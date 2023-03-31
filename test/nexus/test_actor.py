from unittest import TestCase
from test.test_utils import StoreDependentTestCase, ActorDependentTestCase
from improv.actor import Actor, RunManager, AsyncRunManager, Spike
from improv.store import Store
from improv.nexus import AsyncQueue, Link
from multiprocessing import Process
import numpy as np
from pyarrow.lib import ArrowIOError
import pickle
import pyarrow.plasma as plasma
import asyncio
from queue import Empty
import time
from typing import Awaitable, Callable
import traceback
from improv.store import ObjectNotFoundError
import pickle
from queue import Queue
import logging
from logging import warning
import contextlib
import logging; logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


#TODO: write actor unittests
#NOTE: Unittests are getting resourcewarning
#setStore
#NOTE: src.nexus.* should be improv.*

class RunManager_setupRun(ActorDependentTestCase):

    def setUp(self):
        super(RunManager_setupRun, self).setUp()
        self.actor=Actor('test')
        self.isSetUp= False;
        self.runNum=0

    def test_runManager(self):
        q_sig= Queue()
        q_sig.put('setup')
        q_sig.put('run') #runs after this signal
        q_sig.put('pause')
        q_sig.put('resume') #runs again after this signal
        q_sig.put('quit')
        q_comm= Queue()
        with RunManager('test', self.runMethod, self.run_setup, q_sig, q_comm) as rm:
            print(rm)
        self.assertEqual(self.runNum, 2)
        self.assertTrue(self.isSetUp)

    def tearDown(self):
        super(RunManager_setupRun, self).tearDown()

class RunManager_process(ActorDependentTestCase):
    def setUp(self):
        super(RunManager_process, self).setUp()
        self.actor=Actor('test')

    def test_run(self):
        self.q_sig= Link('queue', 'self', 'process')
        self.q_comm=Link('queue', 'process', 'self')
        self.p2 = Process(target= self.createprocess, args= (self.q_sig, self.q_comm,))
        self.p2.start()
        logger.warning('Entered')
        self.q_sig.put('setup')
        self.assertEqual(self.q_comm.get(), ['ready'])
        self.q_sig.put('run')
        self.assertEqual(self.q_comm.get(), 'ran')
        self.q_sig.put('pause')
        self.q_sig.put('resume')
        self.q_sig.put('quit')
        with self.assertLogs() as cm:
            logging.getLogger().warning('Received pause signal, pending...')
            logging.getLogger().warning('Received resume signal, resuming')
            logging.getLogger().warning('Received quit signal, aborting')
        self.assertEqual(cm.output, ['WARNING:root:Received pause signal, pending...',
        'WARNING:root:Received resume signal, resuming', 'WARNING:root:Received quit signal, aborting'])
        self.p2.join()
        self.p2.terminate()

    def tearDown(self):
        super(RunManager_process, self).tearDown()


'''
Place different actors in separate processes and ensure that run manager is receiving
signals in the expected order.
'''
class AsyncRunManager_Process(ActorDependentTestCase):
    def setUp(self):
        super(AsyncRunManager_Process, self).setUp()

        self.q_sig= Link('queue', 'self', 'process')
        self.q_comm=Link('queue', 'process', 'self')

    async def test_run(self):

        #self.p2 = asyncio.create_subprocess_exec(AsyncRunManager, 'test', self.process_run, self.process_setup, stdin=lf.q_sig, stdout=self.q_comm)
        self.p2 = await Process(target= self.createAsyncProcess, args= (self.q_sig, self.q_comm,))
        self.p2.start()
        self.q_sig.put('setup')
        self.assertEqual(self.q_comm.get(), ['ready'])
        self.assertEqual(self.q_comm.get(), 'ran')
        with self.assertLogs() as cm:
            logging.getLogger().warning('Received pause signal, pending...')
            logging.getLogger().warning('Received resume signal, resuming')
            logging.getLogger().warning('Received quit signal, aborting')
        self.assertEqual(cm.output, ['WARNING:root:Received pause signal, pending...',
        'WARNING:root:Received resume signal, resuming', 'WARNING:root:Received quit signal, aborting'])


    def tearDown(self):
        super(AsyncRunManager_Process, self).tearDown()


class AsyncRunManager_setupRun(ActorDependentTestCase):

    def setUp(self):
        super(AsyncRunManager_setupRun, self).setUp()
        self.actor = Actor('test')
        self.isSetUp = False;
        self.q_sig = Link('test_sig','test_start', 'test_end')
        self.q_comm = Link('test_comm','test_start', 'test_end')
        self.runNum=0

    def load_queue(self):
        self.a_put("setup", 0.1)
        self.a_put('setup',0.1)
        self.a_put('run',0.1)
        self.a_put('pause',0.1)
        self.a_put('resume',0.1)
        self.a_put('quit',0.1)

    async def test_asyncRunManager(self):
        await AsyncRunManager('test', self.runMethod, self.run_setup, self.q_sig, self.q_comm)
        self.assertEqual(self.runNum, 2)
        self.assertTrue(self.isSetUp)

    def tearDown(self):
        super(AsyncRunManager_setupRun, self).tearDown()


class AsyncRunManager_MultiActorTest(ActorDependentTestCase):

    def setUp(self):
        super(AsyncRunManager_MultiActorTest, self).setUp()
        self.actor=Actor('test')
        self.isSetUp= False;
        q_sig = Queue()
        self.q_sig = AsyncQueue(q_sig,'test_sig','test_start', 'test_end')
        q_comm = Queue()
        self.q_comm = AsyncQueue(q_comm, 'test_comm','test_start', 'test_end')
        self.runNum=0

    def actor_1(self):
        self.a_put('resume',0.7)
        self.a_put('pause',0.5)
        self.a_put('setup', 0.1)

    def actor_2(self):
        self.a_put('quit',1)
        self.a_put('run', 0.3)

    async def test_asyncRunManager(self):
        await AsyncRunManager('test', self.runMethod, self.run_setup, self.q_sig, self.q_comm)
        self.assertEqual(self.runNum, 2)

    def tearDown(self):
        super(AsyncRunManager_MultiActorTest, self).tearDown()



