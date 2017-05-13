# code for running the relay process and for general process communication
import os
import time
import logging
from collections import deque

import redis

try: import cPickle as pickle
except ImportError: import pickle

logger = logging.getLogger(__name__)

EXP_KEY = 'es:exp'
TASK_ID_KEY = 'es:task_id'
TASK_DATA_KEY = 'es:task_data'
TASK_CHANNEL = 'es:task_channel'
RESULTS_KEY = 'es:results'


def retry_connect(redis_cfg, tries=300, base_delay=4.):
    for i in range(tries):
        try:
            r = redis.StrictRedis(**redis_cfg)
            r.ping()
            return r
        except redis.ConnectionError as e:
            if i == tries - 1:
                raise
            else:
                delay = base_delay * (1 + (os.getpid() % 10) / 9)
                logger.warning('Could not connect to {}. Retrying after {:.2f} sec ({}/{}). Error: {}'.
                               format(redis_cfg, delay, i + 2, tries, e))
                time.sleep(delay)


def retry_get(pipe, key, tries=300, base_delay=4.):
    # what are vals?
    for i in range(tries):
        # Try to (m)get
        if isinstance(key, (list, tuple)):
            vals = pipe.mget(key)   # pipe?
            if all(v is not None for v in vals):
                return vals
        else:
            val = pipe.get(key)
            if val is not None:
                return val
        # Sleep and retry if any key wasn't available
        if i != tries - 1:
            delay = base_delay * (1 + (os.getpid() % 10) / 9)
            logger.warning('{} not set. Retrying after {:.2f} sec ({}/{})'.format(key, delay, i + 2, tries))
            time.sleep(delay)
    raise RuntimeError('{} not set'.format(key))


class Relay:
    """
    Receives and stores task broadcasts from the master
    Batches and pushes results from worker to master
    """
    def __init__(self, master_redis_cfg, relay_redis_cfg):
        self.master_redis = retry_connect(master_redis_cfg)
        logger.info('[relay] Connected to master: {}'.format(self.master_redis))
        self.local_redis = retry_connect(relay_redis_cfg)
        logger.info('[relay] Connected to relay: {}'.format(self.local_redis))

    def run(self):
        # Initialization: read exp and latest task from master
        self.local_redis.set(EXP_KEY, retry_get(self.master_redis, EXP_KEY))
        self._declare_task_local(*retry_get(self.master_redis, (TASK_ID_KEY, TASK_DATA_KEY)))

        # Start subscribing to tasks
        p = self.master_redis.pubsub()   # TODO: study Redis publishing/subscribing messsaging paradigm
        p.subscribe(**{TASK_CHANNEL: lambda msg: self._declare_task_local(*pickle.loads(msg['data']))})
        p.run_in_thread(sleep_time=0.001)

        # Loop on RESULTS_KEY and push to master
        batch_sizes, last_print_time = deque(maxlen=20), time.time()   # for logging
        while True:
            results = []
            start_time = curr_time = time.time()
            while curr_time - start_time < 0.001:
                results.append(self.local_redis.blpop(RESULTS_KEY)[1])
                curr_time = time.time()
            self.master_redis.rpush(RESULTS_KEY, *results)
            # Log
            batch_sizes.append(len(results))
            if curr_time - last_print_time > 5.0:
                logger.info('[relay] Average batch size {:.3f}'.format(sum(batch_sizes) / len(batch_sizes)))
                last_print_time = curr_time

    def _declare_task_local(self, task_id, task_data):   # why does this make the task local?
        logger.info('[relay] Received task {}'.format(task_id))
        self.local_redis.mset({TASK_ID_KEY: task_id, TASK_DATA_KEY: task_data})

