# Code for running the master process
import logging
import json
import time
import sys
import os
import numpy as np
from pprint import pformat

try: import cPickle as pickle
except ImportError: import pickle
import click

from .relay import retry_connect
from .exp_util import RunningStat, SharedNoiseTable
import exp_util as eu

#logger = logging.getLogger(__name__)

EXP_KEY = 'es:exp'
TASK_ID_KEY = 'es:task_id'
TASK_DATA_KEY = 'es:task_data'
TASK_CHANNEL = 'es:task_channel'
RESULTS_KEY = 'es:results'


# @click.group()
# def cli():
#     logging.basicConfig(
#         format='[%(asctime)s pid=$(process)d] %(message)s',
#         level=logging.INFO,
#         stream=sys.stderr)


# @cli.command()
# @click.option('--exp_file')
# @click.option('--master_socket_path', required=True)
# @click.option('--log_dir')
def start_master(exp_file, master_socket_path, log_dir):
    with open(exp_file, 'r') as f: exp = json.loads(f.read())  # open experiment config file
    log_dir = os.path.expanduser(log_dir) if log_dir else '/tmp/es_master_{}'.format(os.getpid())
    if not os.path.exists(log_dir): os.makedirs(log_dir)  # set up log directory
    run_master({'unix_socket_path': master_socket_path}, log_dir, exp)  # Start the master


def run_master(master_redis_config, log_dir, exp):
    logging.info('run_master: {}'.format(locals()))
    from .optimizers import SGD, Adam
    from . import tabular_logger as tlogger
    logging.info('Tabular logging to {}'.format(log_dir))
    tlogger.start(log_dir)
    config, env, sess, policy = eu.setup(exp, single_threaded=False)
    master = Master(master_redis_config)
    optimizer = {'sgd': SGD, 'adam': Adam}[exp['optimizer']['type']](policy, **exp['optimizer']['args'])
    noise = SharedNoiseTable()  # TODO: figure out more efficient way to do this
    rs = np.random.RandomState()
    ob_stat = RunningStat(
        env.observation_space.shape,
        eps=1e-2  # eps to prevent dividing by zero at the beginning when computing mean/stdev
    )
    #if 'init_from' in exp['policy']:
    #    logger.info('Initializing weights from {}'.format(exp['policy']['init_from']))
    #    policy.initialize_from(exp['policy']['init_from'], ob_stat)

    episodes_so_far = 0
    timesteps_so_far = 0
    tstart = time.time()
    master.declare_experiment(exp)

    while True:
        step_tstart = time.time()
        theta = policy.get_trainable_flat()  # theta = policy parameters?
        assert theta.dtype == np.float32

        curr_task_id = master.declare_task(eu.Task(  # what exactly is a task in this context?
            params=theta,
            ob_mean=ob_stat.mean if policy.needs_ob_stat else None,
            ob_std=ob_stat.std if policy.needs_ob_stat else None,
        ))
        tlogger.log('********** Iteration {} **********'.format(curr_task_id))

        # Pop off results for the current task
        curr_task_results, eval_rets, eval_lens, worker_ids = [], [], [], []
        num_results_skipped, num_episodes_popped, ob_count_this_batch = 0, 0, 0
        while num_episodes_popped < config.episodes_per_batch:
            # Wait for a result
            task_id, result = master.pop_result()
            assert isinstance(task_id, int) and isinstance(result, eu.Result)
            assert (result.eval_return is None) == (result.eval_length is None)  # must either be both T or both F?
            worker_ids.append(result.worker_id)

            if result.eval_length is not None:
                # This was an eval job
                episodes_so_far += 1
                timesteps_so_far += result.eval_length
                # Store the result only for the current task
                if task_id == curr_task_id:
                    eval_rets.append(result.eval_return)
                    eval_lens.append(result.eval_length)
            else:
                # The real shit
                assert (result.noise_inds_n.ndim == 1 and
                        result.returns_n2.shape == result.lengths_n2.shape == (len(result.noise_inds_n), 2))
                assert result.returns_n2.dtype == np.float32
                # Update counts
                result_num_eps = result.lengths_n2.size
                result_num_timesteps = result.lengths_n2.sum()
                episodes_so_far += result_num_eps
                timesteps_so_far += result_num_timesteps
                # Store results only for current tasks
                if task_id == curr_task_id:
                    curr_task_results.append(result)
                    num_episodes_popped += result_num_eps
                    # Update ob stats
                    if policy.needs_ob_stat and result.ob_count > 0:
                        ob_stat.increment(result.ob_sum, result.ob_sumsq, result.ob_count)
                        ob_count_this_batch += result.ob_count
                else:
                    num_results_skipped += 1

        # Compute skip fraction
        frac_results_skipped = num_results_skipped / (num_results_skipped + len(curr_task_results))
        if num_results_skipped > 0:
            logging.warning('Skipped {} out of date results ({:.2f}%)'.format(
                num_results_skipped, 100. * frac_results_skipped))

        # Assemble results
        noise_inds_n = np.concatenate([r.noise_inds_n for r in curr_task_results])
        returns_n2 = np.concatenate([r.returns_n2 for r in curr_task_results])
        lengths_n2 = np.concatenate([r.lengths_n2 for r in curr_task_results])
        assert noise_inds_n.shape[0] == returns_n2.shape[0] == lengths_n2.shape[0]
        # Process returns
        if config.return_proc_mode == 'centered_rank':
            proc_returns_n2 = eu.compute_centered_ranks(returns_n2)
        elif config.return_proc_mode == 'sign':  # the signs of the results? <-- that seems wrong
            proc_returns_n2 = np.concatenate([r.signreturns_n2 for r in curr_task_results])
        elif config.return_proc_mode == 'centered_sign_rank':
            proc_returns_n2 = eu.compute_centered_ranks(np.concatenate([r.signreturns_n2 for r in curr_task_results]))
        else:
            raise NotImplementedError(config.return_proc_mode)
        # Compute and take step
        g, count = eu.batched_weighted_sum(
            proc_returns_n2[:, 0] - proc_returns_n2[:, 1],
            (noise.get(idx, policy.num_params) for idx in noise_inds_n),
            batch_size=500
        )
        g /= returns_n2.size
        assert g.shape == (policy.num_params,) and g.dtype == np.float32 and count == len(noise_inds_n)
        update_ratio = optimizer.update(-g + config.l2coeff * theta)

        # Update ob stat (we're never running the policy in the master, but we might be snapshotting the policy)
        if policy.needs_ob_stat:
            policy.set_ob_stat(ob_stat.mean, ob_stat.std)

        step_tend = time.time()
        tlogger.record_tabular("EpRewMean", returns_n2.mean())
        tlogger.record_tabular("EpRewStd", returns_n2.std())
        tlogger.record_tabular("EpLenMean", lengths_n2.mean())

        tlogger.record_tabular("EvalEpRewMean", np.nan if not eval_rets else np.mean(eval_rets))
        tlogger.record_tabular("EvalEpRewStd", np.nan if not eval_rets else np.std(eval_rets))
        tlogger.record_tabular("EvalEpLenMean", np.nan if not eval_rets else np.mean(eval_lens))
        tlogger.record_tabular("EvalPopRank", np.nan if not eval_rets else (
            np.searchsorted(np.sort(returns_n2.ravel()), eval_rets).mean() / returns_n2.size))  # ???
        tlogger.record_tabular("EvalEpCount", len(eval_rets))

        tlogger.record_tabular("Norm", float(np.square(policy.get_trainable_flat()).sum()))
        tlogger.record_tabular("GradNorm", float(np.square(g).sum()))
        tlogger.record_tabular("UpdateRatio", float(update_ratio))

        tlogger.record_tabular("EpisodesThisIter", lengths_n2.size)
        tlogger.record_tabular("EpisodesSoFar", episodes_so_far)
        tlogger.record_tabular("TimestepsThisIter", lengths_n2.sum())
        tlogger.record_tabular("TimestepsSoFar", timesteps_so_far)

        num_unique_workers = len(set(worker_ids))
        tlogger.record_tabular("UniqueWorkers", num_unique_workers)
        tlogger.record_tabular("UniqueWorkersFrac", num_unique_workers / len(worker_ids))
        tlogger.record_tabular("ResultsSkippedFrac", frac_results_skipped)
        tlogger.record_tabular("ObCount", ob_count_this_batch)

        tlogger.record_tabular("TimeElapsedThisIter", step_tend - step_tstart)
        tlogger.record_tabular("TimeElapsed", step_tend - tstart)
        tlogger.dump_tabular()

        # save a snapshot of the policy?
        if config.snapshot_freq != 0 and curr_task_id % config.snapshot_freq == 0:
            import os.path as osp
            filename = osp.join(tlogger.get_dir(), 'snapshot_iter{:05d}_rew{}.h5'.format(
                curr_task_id,
                np.nan if not eval_rets else int(np.mean(eval_rets))
            ))
            assert not osp.exists(filename)
        policy.save(filename)
        tlogger.log('Saved snapshot {}'.format(filename))


class Master:
    def __init__(self, master_redis_config):
        self.task_counter = 0   # for keeping track of number of workers running
        self.master_redis = retry_connect(master_redis_config)
        logging.info('[master] Connected to Redis {}'.format(self.master_redis))

    def declare_experiment(self, exp):
        self.master_redis.set(EXP_KEY, pickle.dumps(exp, protocol=-1))   # storing the experiment
        logging.info('[master] Declared experiment {}'.format(pformat(exp)))

    def declare_task(self, task_data):   # pass data and task to new worker?
        task_id = self.task_counter
        self.task_counter += 1

        # send serialized task data to worker?
        serialized_task_data = pickle.dumps(task_data, protocol=-1)
        (self.master_redis.pipeline()
         .mset({TASK_ID_KEY: task_id, TASK_DATA_KEY: serialized_task_data})
         .publish(TASK_CHANNEL, pickle.dumps((task_id, serialized_task_data), protocol=-1))
         .execute())
        logging.debug('[master] Declared task {}'.format(task_id))
        return task_id

    def pop_result(self):
        # get task data and result from most recently returned worker
        task_id, result = pickle.loads(self.master_redis.blpop(RESULTS_KEY)[1])
        logging.debug('[master] Popped a result for task {}'.format(task_id))
        return task_id, result