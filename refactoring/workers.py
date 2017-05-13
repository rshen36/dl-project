# Code for running the worker processes
import os
import time
import logging
import multiprocessing
import numpy as np

try: import cPickle as pickle
except ImportError: import pickle

from .relay import Relay, retry_connect, retry_get
from exp_util import SharedNoiseTable, RunningStat
import exp_util as eu
import redis

logger = logging.getLogger(__name__)

EXP_KEY = 'es:exp'
TASK_ID_KEY = 'es:task_id'
TASK_DATA_KEY = 'es:task_data'
TASK_CHANNEL = 'es:task_channel'
RESULTS_KEY = 'es:results'


#@cli.command()
def start_workers(master_socket_path, relay_socket_path, num_workers):
    master_redis_cfg = {'unix_socket_path': master_socket_path}
    relay_redis_cfg = {'unix_socket_path': relay_socket_path}
    if os.fork() == 0:
        Relay(master_redis_cfg, relay_redis_cfg).run()  # Start the relay
        return
    # Start the workers
    noise = SharedNoiseTable()  # Workers share the same noise
    num_workers = num_workers if num_workers else multiprocessing.cpu_count()
    logging.info('Spawning {} workers'.format(num_workers))
    for _ in range(num_workers):
        if os.fork() == 0:
            run_worker(relay_redis_cfg, noise=noise)
            return
    os.wait()


def run_worker(relay_redis_cfg, noise, min_task_runtime=.2):
    logger.info('run_worker: {}'.format(locals()))
    assert isinstance(noise, SharedNoiseTable)
    worker = Worker(relay_redis_cfg)
    exp = worker.get_experiment()
    config, env, sess, policy = eu.setup(exp, single_threaded=True)
    rs = np.random.RandomState()
    worker_id = rs.randint(2 ** 31)  # don't have to order worker_ids and guarantee no two same id?

    assert policy.needs_ob_stat == (config.calc_obstat_prob != 0)

    while True:
        task_id, task_data = worker.get_current_task()
        task_tstart = time.time()
        assert isinstance(task_id, int) and isinstance(task_data, eu.Task)
        if policy.needs_ob_stat:
            policy.set_ob_stat(task_data.ob_mean, task_data.ob_std)

        if rs.rand() < config.eval_prob:
            # Evaluation: noiseless weights and noiseless actions
            policy.set_trainable_flat(task_data.params)
            eval_rews, eval_length = policy.rollout(env)
            eval_return = eval_rews.sum()
            logger.info('Eval result: task={} return={:3f} length={}'.format(task_id, eval_return, eval_length))
            worker.push_result(task_id, eu.Result(
                worker_id=worker_id,
                noise_inds_n=None,
                returns_n2=None,
                signreturns_n2=None,
                lengths_n2=None,
                eval_return=eval_return,
                eval_length=eval_length,
                ob_sum=None,
                ob_sumsq=None,
                ob_count=None
            ))
        else:
            # Rollouts with noise
            noise_inds, returns, signreturns, lengths = [], [], [], []
            task_ob_stat = RunningStat(env.observation_space.shape, eps=0.)  # eps=0 bc only incrementing

            while not noise_inds or time.time() - task_tstart < min_task_runtime:
                noise_idx = noise.sample_index(rs, policy.num_params)  # sampling indices for random noise values?
                v = config.noise_stdev * noise.get(noise_idx, policy.num_params)

                # finite differences (adding and subtracting v)?
                policy.set_trainable_flat(task_data.params + v)
                rews_pos, len_pos = rollout_and_update_ob_stat(
                    policy, env, rs, task_ob_stat, config.calc_obstat_prob)

                policy.set_trainable_flat(task_data.params - v)
                rews_neg, len_neg = rollout_and_update_ob_stat(
                    policy, env, rs, task_ob_stat, config.calc_obstat_prob)

                # want to keep track of sign returns bc of way fd calculated?
                noise_inds.append(noise_idx)
                returns.append([rews_pos.sum(), rews_neg.sum()])
                signreturns.append([np.sign(rews_pos).sum(), np.sign(rews_neg).sum()])
                lengths.append([len_pos, len_neg])

            worker.push_result(task_id, eu.Result(
                worker_id=worker_id,
                noise_inds_n=np.array(noise_inds),
                returns_n2=np.array(returns, dtype=np.float32),
                signreturns_n2=np.array(signreturns, dtype=np.float32),
                lengths_n2=np.array(lengths, dtype=np.int32),
                eval_return=None,
                eval_length=None,
                ob_sum=None if task_ob_stat.count == 0 else task_ob_stat.sum,
                ob_sumsq=None if task_ob_stat.count == 0 else task_ob_stat.sumsq,
                ob_count=task_ob_stat.count
            ))


# TODO: figure out how to get rendering to work
def rollout_and_update_ob_stat(policy, env, rs, task_ob_stat, calc_obstat_prob):
    if policy.needs_ob_stat and calc_obstat_prob != 0 and rs.rand() < calc_obstat_prob:   # why a probability?
        rollout_rews, rollout_len, obs = policy.rollout(env, save_obs=True, random_stream=rs)
        task_ob_stat.increment(obs.sum(axis=0), np.square(obs).sum(axis=0), len(obs))
    else:
        rollout_rews, rollout_len = policy.rollout(env, random_stream=rs)
    return rollout_rews, rollout_len


class Worker:
    def __init__(self, relay_redis_cfg):
        self.local_redis = retry_connect(relay_redis_cfg)
        logger.info('[worker] Connected to relay: {}'.format(self.local_redis))

        self.cached_task_id, self.cached_task_data = None, None

    def get_experiment(self):
        # Grab experiment info
        exp = pickle.loads(retry_get(self.local_redis, EXP_KEY))
        logger.info('[worker] Experiment: {}'.format(exp))
        return exp

    # TODO: study redis pipelines
    def get_current_task(self):
        with self.local_redis.pipeline() as pipe:
            while True:
                try:
                    pipe.watch(TASK_ID_KEY)
                    task_id = int(retry_get(pipe, TASK_ID_KEY))
                    if task_id == self.cached_task_id:
                        logger.debug('[worker] Returned cached task {}'.format(task_id))
                        break
                    pipe.multi()   # multi?
                    pipe.get(TASK_DATA_KEY)
                    logger.info('[worker] Getting new task {}. Cached task was {}'.
                                format(task_id, self.cached_task_id))
                    self.cached_task_id, self.cached_task_data = task_id, pickle.loads(pipe.execute()[0])
                    break
                except redis.WatchError:
                    continue
        return self.cached_task_id, self.cached_task_data

    def push_result(self, task_id, result):
        self.local_redis.rpush(RESULTS_KEY, pickle.dunmps((task_id, result), protocol=-1))
        logger.debug('[worker] Pushed result for task {}'.format(task_id))