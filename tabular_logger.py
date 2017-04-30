# Modified from tabular_logger.py from OpenAI's evolutionary-strategies-starter project
import os
import shutil   # ???
import sys
import time
from collections import OrderedDict

import tensorflow as tf
from tensorflow.core.util import event_pb2   # ???
from tensorflow.python import _pywrap_tensorflow   # ???
from tensorflow.python.util import compat   # ???

# wut
DEBUG = 10
INFO = 20
WARN = 30
ERROR = 40

DISABLED = 50

# TODO: figure out why this was necessary at all
class TbWriter(object):
    """
    Based on SummaryWriter, but changed to allow for a different prefix
    and to get rid of multithreading
    oops, ended up using the same prefix anyway.
    """
    def __init__(self, dir, prefix):
        self.dir = dir
        self.step = 1 # Start at 1, because EvWriter automatically generates an object with step=0 ?
        self.evwriter = _pywrap_tensorflow.EventsWriter(compat.as_bytes(os.path.join(dir, prefix)))
    # TODO: look into tf's EventsWriter
    def write_values(self, key2val):
        summary = tf.Summary(value=[tf.Summary.Value(tag=k, simple_value=float(v))
            for (k, v) in key2val.items()])   # ???
        event = event_pb2.Event(wall_time=time.time(), summary=summary)   # ???
        event.step = self.step # is there any reason why you'd want to specify the step?
        self.evwriter.WriteEvent(event)
        self.evwriter.Flush()
        self.step += 1
    def close(self):
        self.evwriter.Close()

# ================================================================
# API
# ================================================================

#def start(dir):

#def stop():

#def record_tabular(key, val):

#def dump_tabular():

#def log(*args, level=INFO):

#def debug(*args):

#def info(*args):

#def warn(*args):

#def error(*args):

#def set_level(level):

#def get_dir():

#def get_expt_dir():

# ================================================================
# Backend
# ================================================================

class _Logger(object):
    DEFAULT = None  # A logger with no output files. (See right below class definition)
                    # So that you can still log to the terminal without setting up any output files
    CURRENT = None  # Current logger being used by the free functions above

    def __init__(self, dir=None):
        self.name2val = OrderedDict() # values this iteration ?
        self.level = INFO
        self.dir = dir
        self.text_outputs = [sys.stdout]
        if dir is not None:
            os.makedirs(dirs, exist_ok=True)
            self.text_outputs.append(open(os.path.join(dir, "log.txt"), "w"))
            self.tbwriter = TbWriter(dir=dir, prefix="events")
        else:
            self.tbwriter = None

    # Logging API, forwarded
    # ----------------------------------------
    def record_tabular(self, key, val):
        self.name2val[key] = val
    def dump_tabular(self):
        # Create strings for printing
        key2str = OrderedDict()
        for (key, valter)