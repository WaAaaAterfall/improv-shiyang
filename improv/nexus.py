import os
import uuid
import signal
import logging
import asyncio
import concurrent
import subprocess

from queue import Full
from datetime import datetime
from multiprocessing import Process, get_context
from importlib import import_module

import zmq.asyncio as zmq
from zmq import PUB, REP, SocketOption

from improv.store import StoreInterface
from improv.actor import Signal
from improv.config import Config
from improv.link import Link, MultiLink

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# TODO: Set up store.notify in async function (?)


class Nexus:
    """Main server class for handling objects in improv"""

    def __init__(self, name="Server"):
        self.name = name

    def __str__(self):
        return self.name

    def createNexus(
        self,
        file=None,
        use_hdd=False,
        use_watcher=False,
        store_size=10000000,
        control_port=0,
        output_port=0,
    ):
        """Start the server, the store, and load the config file.
        This will also create the actors.
        """

        curr_dt = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        logger.info(f"************ new improv server session {curr_dt} ************")

        # set up socket in lieu of printing to stdout
        self.zmq_context = zmq.Context()
        self.out_socket = self.zmq_context.socket(PUB)
        self.out_socket.bind("tcp://*:%s" % output_port)
        out_port_string = self.out_socket.getsockopt_string(SocketOption.LAST_ENDPOINT)
        output_port = int(out_port_string.split(":")[-1])

        self.in_socket = self.zmq_context.socket(REP)
        self.in_socket.bind("tcp://*:%s" % control_port)
        in_port_string = self.in_socket.getsockopt_string(SocketOption.LAST_ENDPOINT)
        control_port = int(in_port_string.split(":")[-1])

        # default size should be system-dependent; this is 40 GB
        self._startStoreInterface(store_size)
        self.out_socket.send_string("StoreInterface started")

        # connect to store and subscribe to notifications
        logger.info("Create new store object")
        self.store = StoreInterface(store_loc=self.store_loc)
        self.store.subscribe()

        # LMDB storage
        self.use_hdd = use_hdd
        if self.use_hdd:
            self.lmdb_name = f'lmdb_{datetime.now().strftime("%Y%m%d_%H%M%S")}'
            self.store_dict = dict()

        # TODO: Better logic/flow for using watcher as an option
        self.p_watch = None
        if use_watcher:
            self.startWatcher()

        # Create dicts for reading config and creating actors
        self.comm_queues = {}
        self.sig_queues = {}
        self.data_queues = {}
        self.actors = {}
        self.flags = {}
        self.processes = []

        if file is None:
            logger.exception("Need a config file!")
            raise Exception  # TODO
        else:
            self.loadConfig(file=file)

        self.flags.update({"quit": False, "run": False, "load": False})
        self.allowStart = False
        self.stopped = False

        return (control_port, output_port)

    def loadConfig(self, file):
        """For each connection:
        create a Link with a name (purpose), start, and end
        Start links to one actor's name, end to the other.
        Nexus gives start_actor the Link as a q_in,
          and end_actor the Link as a q_out
        Nexus maintains dict of name and associated Link.
        Nexus also has list of Links that it is itself connected to
          for communication purposes.
        OR
        For each connection, create 2 Links. Nexus acts as intermediary.
        """
        # TODO load from file or user input, as in dialogue through FrontEnd?

        self.config = Config(configFile=file)
        flag = self.config.createConfig()
        if flag == -1:
            logger.error(
                "An error occurred when loading the configuration file. "
                "Please see the log file for more details."
            )

        # create all data links requested from Config config
        self.createConnections()

        if self.config.hasGUI:
            # Have to load GUI first (at least with Caiman)
            name = self.config.gui.name
            m = self.config.gui  # m is ConfigModule
            # treat GUI uniquely since user communication comes from here
            try:
                visualClass = m.options["visual"]
                # need to instantiate this actor
                visualActor = self.config.actors[visualClass]
                self.createActor(visualClass, visualActor)
                # then add links for visual
                for k, l in {
                    key: self.data_queues[key]
                    for key in self.data_queues.keys()
                    if visualClass in key
                }.items():
                    self.assignLink(k, l)

                # then give it to our GUI
                self.createActor(name, m)
                self.actors[name].setup(visual=self.actors[visualClass])

                self.p_GUI = Process(target=self.actors[name].run, name=name)
                self.p_GUI.daemon = True
                self.p_GUI.start()

            except Exception as e:
                logger.error(f"Exception in setting up GUI {name}: {e}")

        else:
            # have fake GUI for communications
            q_comm = Link("GUI_comm", "GUI", self.name)
            self.comm_queues.update({q_comm.name: q_comm})

        # First set up each class/actor
        for name, actor in self.config.actors.items():
            if name not in self.actors.keys():
                # Check for actors being instantiated twice
                try:
                    self.createActor(name, actor)
                    logger.info(f"Setting up actor {name}")
                except Exception as e:
                    logger.error(f"Exception in setting up actor {name}: {e}.")
                    self.quit()

        # Second set up each connection b/t actors
        # TODO: error handling for if a user tries to use q_in without defining it
        for name, link in self.data_queues.items():
            self.assignLink(name, link)

        if self.config.settings["use_watcher"] is not None:
            watchin = []
            for name in self.config.settings["use_watcher"]:
                watch_link = Link(name + "_watch", name, "Watcher")
                self.assignLink(name + ".watchout", watch_link)
                watchin.append(watch_link)
            self.createWatcher(watchin)

    def startNexus(self):
        """
        Puts all actors in separate processes and begins polling
            to listen to comm queues
        """
        for name, m in self.actors.items():
            if "GUI" not in name:  # GUI already started
                if "method" in self.config.actors[name].options:
                    meth = self.config.actors[name].options["method"]
                    logger.info("This actor wants: {}".format(meth))
                    ctx = get_context(meth)
                    p = ctx.Process(target=m.run, name=name)
                else:
                    ctx = get_context("fork")
                    p = ctx.Process(target=self.runActor, name=name, args=(m,))
                    if "Watcher" not in name:
                        if "daemon" in self.config.actors[name].options:
                            p.daemon = self.config.actors[name].options["daemon"]
                            logger.info("Setting daemon for {}".format(name))
                        else:
                            p.daemon = True  # default behavior
                self.processes.append(p)

        self.start()

        loop = asyncio.get_event_loop()
        try:
            self.out_socket.send_string("Awaiting input:")
            res = loop.run_until_complete(self.pollQueues())
        except asyncio.CancelledError:
            logger.info("Loop is cancelled")

        try:
            logger.info(f"Result of run_until_complete: {res}")
        except Exception as e:
            logger.info(f"Res failed to await: {e}")

        logger.info(f"Current loop: {asyncio.get_event_loop()}")

        loop.stop()
        loop.close()
        logger.info("Shutdown loop")
        self.zmq_context.destroy()

    def start(self):
        logger.info("Starting processes")

        for p in self.processes:
            logger.info(str(p))
            p.start()

        logger.info("All processes started")

    def destroyNexus(self):
        """Method that calls the internal method
        to kill the process running the store (plasma server)
        """
        logger.warning("Destroying Nexus")
        self._closeStoreInterface()

        try:
            os.remove(self.store_loc)
        except FileNotFoundError:
            logger.warning(
                "StoreInterface file {} is already deleted".format(self.store_loc)
            )
        logger.warning("Delete the store at location {0}".format(self.store_loc))

    async def pollQueues(self):
        """
        Listens to links and processes their signals.

        For every communications queue connected to Nexus, a task is
        created that gets from the queue. Throughout runtime, when these
        queues output a signal, they are processed by other functions.
        At the end of runtime (when the gui has been closed), polling is
        stopped.

        Returns:
            "Shutting down" (string): Notifies start() that pollQueues has completed.
        """
        self.actorStates = dict.fromkeys(self.actors.keys())
        if not self.config.hasGUI:
            # Since Visual is not started, it cannot send a ready signal.
            try:
                del self.actorStates["Visual"]
            except Exception as e:
                logger.info("Visual is not started: {0}".format(e))
                pass
        polling = list(self.comm_queues.values())
        pollingNames = list(self.comm_queues.keys())
        self.tasks = []
        for q in polling:
            self.tasks.append(asyncio.create_task(q.get_async()))

        self.tasks.append(asyncio.create_task(self.remote_input()))
        self.early_exit = False

        # add signal handlers
        loop = asyncio.get_event_loop()
        signals = (signal.SIGHUP, signal.SIGTERM, signal.SIGINT)
        for s in signals:
            loop.add_signal_handler(
                s, lambda s=s: self.stop_polling_and_quit(s, polling)
            )

        while not self.flags["quit"]:
            try:
                done, pending = await asyncio.wait(
                    self.tasks, return_when=concurrent.futures.FIRST_COMPLETED
                )
            except asyncio.CancelledError:
                pass

            # sort through tasks to see where we got input from
            # (so we can choose a handler)
            for i, t in enumerate(self.tasks):
                if i < len(polling):
                    if t in done or polling[i].status == "done":
                        # catch tasks that complete await wait/gather
                        r = polling[i].result
                        if r:
                            if "GUI" in pollingNames[i]:
                                self.processGuiSignal(r, pollingNames[i])
                            else:
                                self.processActorSignal(r, pollingNames[i])
                            self.tasks[i] = asyncio.create_task(polling[i].get_async())
                elif t in done:
                    logger.debug("t.result = " + str(t.result()))
                    self.tasks[i] = asyncio.create_task(self.remote_input())

        if not self.early_exit:  # don't run this again if we already have
            self.stop_polling("quit", polling)
            logger.warning("Shutting down polling")
        return "Shutting Down"

    def stop_polling_and_quit(self, signal, queues):
        logger.warn("Shutting down via signal handler for {}".format(signal))
        logger.warn("Steps may be out of order or dirty.")

        self.stop_polling(signal, queues)
        self.flags["quit"] = True
        self.early_exit = True
        self.quit()

    async def remote_input(self):
        msg = await self.in_socket.recv_multipart()
        command = msg[0].decode("utf-8")
        await self.in_socket.send_string("Awaiting input:")
        if command == Signal.quit():
            await self.out_socket.send_string("QUIT")
        self.processGuiSignal([command], "TUI_Nexus")

    def processGuiSignal(self, flag, name):
        """Receive flags from the Front End as user input"""
        name = name.split("_")[0]
        if flag:
            logger.info("Received signal from user: " + flag[0])
            if flag[0] == Signal.run():
                logger.info("Begin run!")
                # self.flags['run'] = True
                self.run()
            elif flag[0] == Signal.setup():
                logger.info("Running setup")
                self.setup()
            elif flag[0] == Signal.ready():
                logger.info("GUI ready")
                self.actorStates[name] = flag[0]
            elif flag[0] == Signal.quit():
                logger.warning("Quitting the program!")
                self.flags["quit"] = True
                self.quit()
            elif flag[0] == Signal.load():
                logger.info("Loading Config config from file " + flag[1])
                self.loadConfig(flag[1])
            elif flag[0] == Signal.pause():
                logger.info("Pausing processes")
                # TODO. Also resume, reset

            # temporary WiP
            elif flag[0] == Signal.kill():
                # TODO: specify actor to kill
                list(self.processes)[0].kill()
            elif flag[0] == Signal.revive():
                dead = [p for p in list(self.processes) if p.exitcode is not None]
                for pro in dead:
                    name = pro.name
                    m = self.actors[pro.name]
                    actor = self.config.actors[name]
                    if "GUI" not in name:  # GUI hard to revive independently
                        if "method" in actor.options:
                            meth = actor.options["method"]
                            logger.info("This actor wants: {}".format(meth))
                            ctx = get_context(meth)
                            p = ctx.Process(target=m.run, name=name)
                        else:
                            ctx = get_context("fork")
                            p = ctx.Process(target=self.runActor, name=name, args=(m,))
                            if "Watcher" not in name:
                                if "daemon" in actor.options:
                                    p.daemon = actor.options["daemon"]
                                    logger.info("Setting daemon for {}".format(name))
                                else:
                                    p.daemon = True

                    # Setting the stores for each actor to be the same
                    # TODO: test if this works for fork -- don't think it does?
                    al = [act for act in self.actors.values() if act.name != pro.name]
                    m.setStoreInterface(al[0].client)
                    m.client = None
                    m._getStoreInterface()

                    self.processes.append(p)
                    p.start()
                    m.q_sig.put_nowait(Signal.setup())
                    # TODO: ensure waiting for ready before run?
                    m.q_sig.put_nowait(Signal.run())

                self.processes = [p for p in list(self.processes) if p.exitcode is None]
            elif flag[0] == Signal.stop():
                logger.info("Nexus received stop signal")
                self.stop()
        elif flag:
            logger.error("Unknown signal received from Nexus: {}".format(flag))

    def processActorSignal(self, sig, name):
        if sig is not None:
            logger.info("Received signal " + str(sig[0]) + " from " + name)
            state_val = self.actorStates.values()
            if not self.stopped and sig[0] == Signal.ready():
                self.actorStates[name.split("_")[0]] = sig[0]
                if all(val == Signal.ready() for val in state_val):
                    self.allowStart = True
                    # TODO: replace with q_sig to FE/Visual
                    logger.info("Allowing start")

            elif self.stopped and sig[0] == Signal.stop_success():
                self.actorStates[name.split("_")[0]] = sig[0]
                if all(val == Signal.stop_success() for val in state_val):
                    self.allowStart = True  # TODO: replace with q_sig to FE/Visual
                    self.stoppped = False
                    logger.info("All stops were successful. Allowing start.")

    def setup(self):
        for q in self.sig_queues.values():
            try:
                logger.info("Starting setup: " + str(q))
                q.put_nowait(Signal.setup())
            except Full:
                logger.warning("Signal queue" + q.name + "is full")

    def run(self):
        if self.allowStart:
            for q in self.sig_queues.values():
                try:
                    q.put_nowait(Signal.run())
                except Full:
                    logger.warning("Signal queue" + q.name + "is full")
                    # queue full, keep going anyway
                    # TODO: add repeat trying as async task
        else:
            logger.error("Not all actors ready yet, please wait and then try again.")

    def quit(self):
        logger.warning("Killing child processes")
        self.out_socket.send_string("QUIT")

        for q in self.sig_queues.values():
            try:
                q.put_nowait(Signal.quit())
            except Full:
                logger.warning("Signal queue {} full, cannot quit".format(q.name))
            except FileNotFoundError:
                logger.warning("Queue {} corrupted.".format(q.name))

        if self.config.hasGUI:
            self.processes.append(self.p_GUI)

        if self.p_watch:
            self.processes.append(self.p_watch)

        for p in self.processes:
            p.terminate()
            p.join()

        logger.warning("Actors terminated")

        self.destroyNexus()

    def stop(self):
        logger.warning("Starting stop procedure")
        self.allowStart = False

        for q in self.sig_queues.values():
            try:
                q.put_nowait(Signal.stop())
            except Full:
                logger.warning("Signal queue" + q.name + "is full")
        self.allowStart = True

    def revive(self):
        logger.warning("Starting revive")

    def stop_polling(self, stop_signal, queues):
        """Cancels outstanding tasks and fills their last request.

        Puts a string into all active queues, then cancels their
        corresponding tasks. These tasks are not fully cancelled until
        the next run of the event loop.

        Args:
            stop_signal (signal.signal): Signal for signal handler.
            queues (AsyncQueue): Comm queues for links.
        """
        logger.info("Received shutdown order")

        logger.info(f"Stop signal: {stop_signal}")
        shutdown_message = "SHUTDOWN"
        for q in queues:
            try:
                q.put(shutdown_message)
            except Exception:
                logger.info("Unable to send shutdown message to {}.".format(q.name))

        logger.info("Canceling outstanding tasks")

        [task.cancel() for task in self.tasks]

        logger.info("Polling has stopped.")

    def createStoreInterface(self, name):
        """Creates StoreInterface w/ or w/out LMDB
        functionality based on {self.use_hdd}."""
        if not self.use_hdd:
            return StoreInterface(name, self.store_loc)
        else:
            if name not in self.store_dict:
                self.store_dict[name] = StoreInterface(
                    name, self.store_loc, use_hdd=True, lmdb_name=self.lmdb_name
                )
            return self.store_dict[name]

    def _startStoreInterface(self, size):
        """Start a subprocess that runs the plasma store
        Raises a RuntimeError exception size is undefined
        Raises an Exception if the plasma store doesn't start

        #TODO: Generalize this to non-plasma stores
        """
        if size is None:
            raise RuntimeError("Server size needs to be specified")
        try:
            self.store_loc = str(os.path.join("/tmp/", str(uuid.uuid4())))
            self.p_StoreInterface = subprocess.Popen(
                [
                    "plasma_store",
                    "-s",
                    self.store_loc,
                    "-m",
                    str(size),
                    "-e",
                    "hashtable://test",
                ],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            logger.info("StoreInterface start successful: {}".format(self.store_loc))
        except Exception as e:
            logger.exception("StoreInterface cannot be started: {}".format(e))

    def _closeStoreInterface(self):
        """Internal method to kill the subprocess
        running the store (plasma sever)
        """
        try:
            self.p_StoreInterface.kill()
            self.p_StoreInterface.wait()
            logger.info("StoreInterface close successful: {}".format(self.store_loc))
        except Exception as e:
            logger.exception("Cannot close store {}".format(e))

    def createActor(self, name, actor):
        """Function to instantiate actor, add signal and comm Links,
        and update self.actors dictionary
        """
        # Instantiate selected class
        mod = import_module(actor.packagename)
        clss = getattr(mod, actor.classname)
        instance = clss(actor.name, self.store_loc, **actor.options)

        if "method" in actor.options.keys():
            # check for spawn
            if "fork" == actor.options["method"]:
                # Add link to StoreInterface store
                store = self.createStoreInterface(actor.name)
                instance.setStoreInterface(store)
            else:
                # spawn or forkserver; can't pickle plasma store
                logger.info("No store for this actor yet {}".format(name))
        else:
            # Add link to StoreInterface store
            store = self.createStoreInterface(actor.name)
            instance.setStoreInterface(store)

        q_comm = Link(actor.name + "_comm", actor.name, self.name)
        q_sig = Link(actor.name + "_sig", self.name, actor.name)
        self.comm_queues.update({q_comm.name: q_comm})
        self.sig_queues.update({q_sig.name: q_sig})
        instance.setCommLinks(q_comm, q_sig)

        # Update information
        self.actors.update({name: instance})

    def runActor(self, actor):
        """Run the actor continually; used for separate processes
        #TODO: hook into monitoring here?
        """
        actor.run()

    def createConnections(self):
        """Assemble links (multi or other)
        for later assignment
        """
        for source, drain in self.config.connections.items():
            name = source.split(".")[0]
            # current assumption is connection goes from q_out to something(s) else
            if len(drain) > 1:  # we need multiasyncqueue
                link, endLinks = MultiLink(name + "_multi", source, drain)
                self.data_queues.update({source: link})
                for i, e in enumerate(endLinks):
                    self.data_queues.update({drain[i]: e})
            else:  # single input, single output
                d = drain[0]
                d_name = d.split(".")  # TODO: check if .anything, if not assume q_in
                link = Link(name + "_" + d_name[0], source, d)
                self.data_queues.update({source: link})
                self.data_queues.update({d: link})

    def assignLink(self, name, link):
        """Function to set up Links between actors
        for data location passing
        Actor must already be instantiated

        #NOTE: Could use this for reassigning links if actors crash?
        """
        classname = name.split(".")[0]
        linktype = name.split(".")[1]
        if linktype == "q_out":
            self.actors[classname].setLinkOut(link)
        elif linktype == "q_in":
            self.actors[classname].setLinkIn(link)
        elif linktype == "watchout":
            self.actors[classname].setLinkWatch(link)
        else:
            self.actors[classname].addLink(linktype, link)

    # TODO: StoreInterface access here seems wrong, need to test
    def startWatcher(self):
        from improv.watcher import Watcher

        self.watcher = Watcher("watcher", self.createStoreInterface("watcher"))
        # store = self.createStoreInterface("watcher") if not self.use_hdd else None
        q_sig = Link("watcher_sig", self.name, "watcher")
        self.watcher.setLinks(q_sig)
        self.sig_queues.update({q_sig.name: q_sig})

        self.p_watch = Process(target=self.watcher.run, name="watcher_process")
        self.p_watch.daemon = True
        self.p_watch.start()
        self.processes.append(self.p_watch)
