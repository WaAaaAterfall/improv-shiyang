import asyncio
import logging
from multiprocessing import Manager, cpu_count, set_start_method
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logging.basicConfig(level=logging.DEBUG,
                    format='%(name)s %(message)s',
                    handlers=[logging.FileHandler("global.log"),
                              logging.StreamHandler()])

def Link(name, start, end, limbo):
    """ Function to construct a queue that Nexus uses for
    inter-process (actor) signaling and information passing.

    A Link has an internal queue that can be synchronous (put, get)
    as inherited from multiprocessing.Manager.Queue
    or asynchronous (put_async, get_async) using async executors.
    
    Args:
        See AsyncQueue constructor

    Returns:
        AsyncQueue: queue for communicating between actors and with Nexus
    """

    m = Manager()
    q = AsyncQueue(m.Queue(maxsize=0), name, start, end, limbo)
    return q

class AsyncQueue(object):
    """ Single-output and asynchronous queue class.

    Attributes:
        queue: 
        real_executor: 
        cancelled_join: boolean
        name:
        start:
        end:
        status:
        result:
        limbo:
        num:
        dict:
    """

    def __init__(self, q, name, start, end, limbo):
        """ Constructor for the queue class.

        Args:
            q (Queue): A queue from the Manager class
            name (str): String description of this queue
            start (str): The producer (input) actor for the queue
            end (str): The consumer (output) actor for the queue
            limbo (improv.store.Limbo): Connection to the store for logging in case of future replay

        #TODO: Rewrite to avoid limbo and use logger files instead
        """
        self.queue = q
        self.real_executor = None
        self.cancelled_join = False

        self.name = name
        self.start = start
        self.end = end

        self.status = 'pending'
        self.result = None
        
        self.limbo = limbo
        self.num = 0

    def getStart(self):
        return self.start

    def getEnd(self):
        return self.end

    @property
    def _executor(self):
        if not self.real_executor:
            self.real_executor = ThreadPoolExecutor(max_workers=cpu_count())
        return self.real_executor

    def __getstate__(self):
        self_dict = self.__dict__
        self_dict['_real_executor'] = None
        return self_dict

    def __getattr__(self, name):
        """_summary_

        Args:
            name (_type_): _description_

        Raises:
            AttributeError: Restricts the available attributes to a specific list. This error is raised
            if a different attribute of the queue is requested.
            #TODO: Don't raise this?

        Returns:
            _type_: _description_
        """
        if name in ['qsize', 'empty', 'full',
                    'get', 'get_nowait', 'close']:
            return getattr(self.queue, name)
        else:
            raise AttributeError("'%s' object has no attribute '%s'" %
                                    (self.__class__.__name__, name))

    def __repr__(self):
        return 'Link '+self.name

    def put(self, item):
        """ Function wrapper for put

        Args:
            item (object): Any item that can be sent through a queue
        """
        self.log_to_limbo(item)
        self.queue.put(item)

    def put_nowait(self, item):
        """ Function wrapper for put without waiting

        Args:
            item (object): Any item that can be sent through a queue
        """
        self.log_to_limbo(item)
        self.queue.put_nowait(item)

    async def put_async(self, item):
        """ Coroutine for an asynchronous put

        It adds the put request to the event loop and awaits.

        Args:
            item (object): Any item that can be sent through a queue

        Returns:
            Awaitable or result of the put
        """
        loop = asyncio.get_event_loop()
        self.log_to_limbo(item) #FIXME: This is blocking
        res = await loop.run_in_executor(self._executor, self.put, item)
        return res

    async def get_async(self):
        """ Coroutine for an asynchronous get

        It adds the get request to the event loop and awaits, setting the status to pending. 
        Once the get has returned, it returns the result of the get and sets its status as done.

        Returns:
            Awaitable or result of the get
        
        Exceptions:
            Explicitly passes any exceptions to not hinder execution.
            Errors are logged with the get_async tag.
        """
        loop = asyncio.get_event_loop()
        self.status = 'pending'
        try:
            self.result = await loop.run_in_executor(self._executor, self.get)
            self.status = 'done'
            return self.result
        except Exception as e:
            logger.exception('Error in get_async: {}'.format(e))
            pass

    def cancel_join_thread(self):
        self._cancelled_join = True
        self._queue.cancel_join_thread()

    def join_thread(self):
        self._queue.join_thread()
        if self._real_executor and not self._cancelled_join:
            self._real_executor.shutdown()

    def log_to_limbo(self, item):
        if self.limbo is not None:
            self.limbo.put(item, f'q__{self.start}__{self.num}')
            self.num += 1


def MultiLink(name, start, end, limbo):
    """ Function to generate links for the multi-output queue case.

    Args:
        See constructor for AsyncQueue or MultiAsyncQueue

    Returns:
        MultiAsyncQueue: Producer end of the queue
        List: AsyncQueues for consumers
    """
    
    m = Manager()

    q_out = []
    for endpoint in end:
        q = AsyncQueue(m.Queue(maxsize=0), name, start, endpoint, limbo=limbo)
        q_out.append(q)

    q = MultiAsyncQueue(m.Queue(maxsize=0), q_out, name, start, end)

    return q, q_out

class MultiAsyncQueue(AsyncQueue):
    """ Extension of AsyncQueue to have multiple endpoints.

    Inherits from AsyncQueue. 
    A single producer queue's 'put' is copied to multiple consumer's queues
    q_in is the producer queue, q_out are the consumer queues.
    
    #TODO: test the async nature of this group of queues
    """

    
    def __init__(self, q_in, q_out, name, start, end):
        self.queue = q_in
        self.output = q_out

        self.real_executor = None
        self.cancelled_join = False

        self.name = name
        self.start = start
        self.end = end[0]
        self.status = 'pending'
        self.result = None

    def __repr__(self):
        return 'MultiLink ' + self.name

    def __getattr__(self, name):
        # Remove put and put_nowait and define behavior specifically
        #TODO: remove get capability?
        if name in ['qsize', 'empty', 'full',
                    'get', 'get_nowait', 'close']:
            return getattr(self.queue, name)
        else:
            raise AttributeError("'%s' object has no attribute '%s'" %
                                    (self.__class__.__name__, name))

    def put(self, item):
        for q in self.output:
            q.put(item)

    def put_nowait(self, item):
        for q in self.output:
            q.put_nowait(item)