import pytest
import asyncio
import improv.tui as tui
import logging 
import zmq.asyncio as zmq
from zmq import PUB, SUB, SUBSCRIBE, REQ, REP
from zmq.log.handlers import PUBHandler

CONTROL_PORT = 5555
OUTPUT_PORT = 5556
LOGGING_PORT = 5557 
SERVER_COUNTER = 0

@pytest.fixture
def ports():
    global SERVER_COUNTER
    yield (CONTROL_PORT + SERVER_COUNTER, OUTPUT_PORT + SERVER_COUNTER, LOGGING_PORT + SERVER_COUNTER)
    SERVER_COUNTER += 3

@pytest.fixture
def logger(ports):
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)  
    zmq_log_handler = PUBHandler('tcp://*:%s' % ports[2])
    logger.addHandler(zmq_log_handler)
    yield logger

@pytest.fixture
async def sockets(ports):
    context = zmq.Context()
    ctrl_socket = context.socket(REP)
    ctrl_socket.bind("tcp://*:%s" % ports[0])
    out_socket = context.socket(PUB)
    out_socket.bind("tcp://*:%s" % ports[1])
    yield (ctrl_socket, out_socket)

@pytest.fixture
async def app(ports):
    mock = tui.TUI(*ports)
    yield mock
    await asyncio.sleep(.2)  # give pending tasks a chance to cancel


async def test_console_panel_receives_broadcast(app, sockets, logger):
    async with app.run_test() as pilot:
        await sockets[1].send_string("received")
        await pilot.pause(0.1)
        console = pilot.app.get_widget_by_id("console")
        console.history[0] == 'received'

async def test_quit_from_socket(app, sockets):
    async with app.run_test() as pilot:
        await sockets[1].send_string("QUIT")
        await pilot.pause(0.1)
        assert not pilot.app._running

async def test_log_panel_receives_logging(app, logger):
    async with app.run_test() as pilot:
        logger.info('test')
        await pilot.pause(0.05)
        log_window = pilot.app.get_widget_by_id("log")
        assert log_window.history[0] == 'INFO'
        assert log_window.history[1].rstrip() == 'test'


async def test_input_box_echoed_to_console(app):
    async with app.run_test() as pilot:
        await pilot.press(*'foo', 'enter')
        console = pilot.app.get_widget_by_id("console")
        assert console.history[0] == 'foo'