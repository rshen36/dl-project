# Modified from es.py from OpenAI's evolutionary-strategies-starter project
import logging
import time
#from collections import namedtuple

import numpy as np

from dist_mod import MasterClient, WorkerClient   # .dist = packages?

logger = logging.getLogger(__name__)

#Config = namedtuple()
#Task = namedtuple()
#Result = namedtuple()


#class RunningStat(object):   # object?


#class SharedNoiseTable(object):


#def compute_ranks(x):


#def compute_centered_ranks(x):


def make_session(single_threaded):   # single threaded?
    import tensorflow as tf
    if not single_threaded:
        return tf.InteractiveSession()
    return tf.InteractiveSession(config=tf.ConfigProto(inter_op_parallelism_threads=1, intra_op_parallelism_threads=1))


#def itergroups(items, group_size):


#def batched_weighted_sum(weights, vecs, batch_size):


def setup(exp, single_threaded):
    import gym
    gym.undo_logger_setup()   # why?
    #from . import policies, tf_util   # ???


    #config = Config(**exp['config'])
    env = gym.make(exp['env_id'])
    sess = make_session(single_threaded=single_threaded)
    #policy =
    #tf_util.initialize()

    #return config, env, sess, policy


def run_master(master_redis_cfg, log_dir, exp):
    logger.info('run_master: {}'.format(locals()))   # locals?
    #from optimizers import SGD, Adam
    #from . import tabular_logger as tlogger
    #logger.info('Tabular logging to {}'.format(log_dir))
    #tlogger.start(log_dir)
    #config, env, sess, policy = setup(exp, single_threaded=False)
    master = MasterClient(master_redis_cfg)
    #optimizer = {'sgd': SGD, 'adam': Adam}[exp['optimizer']['type']](policy, **exp['optimizer']['args'])
    #noise = SharedNoiseTable()
    rs = np.random.RandomState()
    #ob_stat =