# Modified from es.py from OpenAI's evolutionary-strategies-starter project
# Also borrows concepts from Denny Britz's reinforcement-learning project
import os
import logging
import time
from collections import namedtuple

import scipy.signal
import tensorflow as tf
import numpy as np

from es_distributed import tf_util as U
from .dist import MasterClient, WorkerClient

logging.basicConfig(filename='experiment.log', level=logging.INFO)
logger = logging.getLogger(__name__)

Config = namedtuple('Config', [
    'l2coeff', 'noise_stdev', 'episodes_per_batch', 'timesteps_per_batch',
    'calc_obstat_prob', 'eval_prob', 'snapshot_freq',
    'return_proc_mode', 'episode_cutoff_mode',
    'es_a3c_prob', 'switch_freq', 'num_local_steps'
])
Task = namedtuple('Task', ['params', 'ob_mean', 'ob_std', 'a3c'])
Result = namedtuple('Result', [
    'worker_id',
    'noise_inds_n', 'returns_n2', 'signreturns_n2', 'lengths_n2',
    'eval_return', 'eval_length',
    'ob_sum', 'ob_sumsq', 'ob_count'
])
Fetched = namedtuple('Fetched', [
    'worker_id', 'grads',
    'terminal', 'ep_return', 'ep_length',
    'eval_return', 'eval_length'
])
Batch = namedtuple('Batch', ['si', 'a', 'adv', 'r', 'terminal', 'features'])
# make separate es and a3c helper function files?
# TODO: check a3c implementation against exact algorithm
# need to implement global counter and t_max for a3c?


class RunningStat(object):
    def __init__(self, shape, eps):
        self.sum = np.zeros(shape, dtype=np.float32)
        self.sumsq = np.full(shape, eps, dtype=np.float32)
        self.count = eps

    def increment(self, s, ssq, c):
        self.sum += s
        self.sumsq += ssq
        self.count += c

    @property
    def mean(self):
        return self.sum / self.count

    @property
    def std(self):
        return np.sqrt(np.maximum(self.sumsq / self.count - np.square(self.mean), 1e-2))

    def set_from_init(self, init_mean, init_std, init_count):
        self.sum[:] = init_mean * init_count
        self.sumsq[:] = (np.square(init_mean) + np.square(init_std)) * init_count
        self.count = init_count


class SharedNoiseTable(object):   # sharing same noise among all workers
    def __init__(self):
        import ctypes, multiprocessing
        seed = 123

        # may need to adapt this number
        count = 250000000   # 1 gigabyte of 32-bit numbers. Will actually sample 2 gigabytes below.

        logger.info('Sampling {} random numbers with seed {}'.format(count, seed))
        self._shared_mem = multiprocessing.Array(ctypes.c_float, count)   # ???
        self.noise = np.ctypeslib.as_array(self._shared_mem.get_obj())   # ???
        assert self.noise.dtype == np.float32
        self.noise[:] = np.random.RandomState(seed).randn(count)   # 64-bit to 32-bit conversion here
        logger.info('Sampled {} bytes'.format(self.noise.size * 4))

    def get(self, i, dim):
        return self.noise[i:(i+dim)]

    def sample_index(self, stream, dim):
        return stream.randint(0, len(self.noise) - dim + 1)


# es helper functions
def compute_ranks(x):
    """
    Returns ranks in [0, len(x))
    Note: This is different from scipy.stats.rankdata, which returns ranks in [1, len(x)].
    """
    assert x.ndim == 1
    ranks = np.empty(len(x), dtype=int)
    ranks[x.argsort()] = np.arange(len(x))
    return ranks


def compute_centered_ranks(x):
    y = compute_ranks(x.ravel()).reshape(x.shape).astype(np.float32)   # ???
    y /= (x.size - 1)
    y -= .5
    return y


def itergroups(items, group_size):
    assert group_size >= 1
    group = []
    for x in items:
        group.append(x)
        if len(group) == group_size:
            yield tuple(group)
            del group[:]
    if group:
        yield tuple(group)


def batched_weighted_sum(weights, vecs, batch_size):
    total = 0.
    num_items_summed = 0
    for batch_weights, batch_vecs in zip(itergroups(weights, batch_size), itergroups(vecs, batch_size)):
        assert len(batch_weights) == len(batch_vecs) <= batch_size
        total += np.dot(np.asarray(batch_weights, dtype=np.float32), np.asarray(batch_vecs, dtype=np.float32))
        num_items_summed += len(batch_weights)
    return total, num_items_summed


# A3C helper functions
def discount(x, gamma):
    return scipy.signal.lfilter([1], [1, -gamma], x[::-1], axis=0)[::-1]  # ???


def process_rollout(rollout, gamma, lambda_=1.0):
    """
    given a rollout, compute its returns and the advantage
    """
    batch_si = np.asarray(rollout.states)
    batch_a = np.asarray(rollout.actions)
    rewards = np.asarray(rollout.rewards)
    vpred_t = np.asarray(rollout.values + [rollout.r])

    rewards_plus_v = np.asarray(rollout.rewards + [rollout.r])
    batch_r = discount(rewards_plus_v, gamma)[:-1]  # ???
    delta_t = rewards + gamma * vpred_t[1:] - vpred_t[:-1]  # ???
    # this formula for the advantage comes from "Generalized Advantage Estimation":
    # https://arxiv.org/abs/1506.02438
    batch_adv = discount(delta_t, gamma * lambda_)

    features = rollout.features[0]  # ???
    return Batch(batch_si, batch_a, batch_adv, batch_r, rollout.terminal, features)


def make_session(single_threaded):
    if not single_threaded:
        return tf.InteractiveSession()
    return tf.InteractiveSession(config=tf.ConfigProto(inter_op_parallelism_threads=1,
                                                       intra_op_parallelism_threads=1))


def setup(exp, single_threaded):
    import gym
    gym.undo_logger_setup()   # to allow for own logging
    from es_distributed import tf_util
    from es_distributed import policies
    #limit = False  # TODO: limit action spaces for Pong and Breakout?

    config = Config(**exp['config'])
    env = gym.make(exp['env_id'])
    # if exp['env_id'] == "Pong-v0" or exp['env_id'] == "Breakout-v0":  # recommendation from Denny Britz
    #     limit = True
    sess = make_session(single_threaded=single_threaded)
    policy = getattr(policies, exp['policy']['type'])(env.observation_space,
                                                      env.action_space, **exp['policy']['args'])
    tf_util.initialize()

    return config, env, sess, policy


def run_master(master_redis_cfg, log_dir, exp):
    # locals(): returns dictionary containing current local symbol table
    logger.info('run_master: {}'.format(locals()))
    from .optimizers import SGD, Adam
    from es_distributed import tabular_logger as tlogger
    logger.info('Tabular logging to {}'.format(log_dir))
    tlogger.start(log_dir)
    config, env, sess, policy = setup(exp, single_threaded=False)
    master = MasterClient(master_redis_cfg)
    optimizer = {'sgd': SGD, 'adam': Adam}[exp['optimizer']['type']](policy, **exp['optimizer']['args'])
    noise = SharedNoiseTable()
    rs = np.random.RandomState()
    ob_stat = RunningStat(
        env.observation_space.shape,
        eps=1e-2   # eps to prevent dividing by zero at the beginning when computing mean/stdev
    )
    # if 'init_from' in exp['policy']:
    #     logger.info('Initializing weights from {}'.format(exp['policy']['init_from']))
    #     policy.initialize_from(exp['policy']['init_from'], ob_stat)

    updates_so_far = 0  # for a3c  <-- probably wrong way to do it
    episodes_so_far = 0
    timesteps_so_far = 0
    tstart = time.time()
    master.declare_experiment(exp)
    a3c = rs.rand() < config.es_a3c_prob

    while True:
        step_tstart = time.time()
        theta = policy.get_trainable_flat()
        assert theta.dtype == np.float32

        curr_task_id = master.declare_task(Task(
            params=theta,
            ob_mean=ob_stat.mean if policy.needs_ob_stat else None,
            ob_std=ob_stat.std if policy.needs_ob_stat else None,
            a3c=a3c
        ))
        tlogger.log('********** Iteration {} **********'.format(curr_task_id))

        if not a3c:  # ES master step
            # Pop off results for the current task
            curr_task_results, eval_rets, eval_lens, worker_ids = [], [], [], []
            num_results_skipped, num_episodes_popped, num_timesteps_popped, ob_count_this_batch = 0, 0, 0, 0
            while num_episodes_popped < config.episodes_per_batch or num_timesteps_popped < config.timesteps_per_batch:
                # Wait for a result
                task_id, result = master.pop_result()
                assert isinstance(task_id, int) and isinstance(result, Result)
                assert (result.eval_return is None) == (result.eval_length is None)
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
                        num_timesteps_popped += result_num_timesteps
                        # Update ob stats
                        if policy.needs_ob_stat and result.ob_count > 0:
                            ob_stat.increment(result.ob_sum, result.ob_sumsq, result.ob_count)
                            ob_count_this_batch += result.ob_count
                    else:
                        num_results_skipped += 1

            # Compute skip fraction
            frac_results_skipped = num_results_skipped / (num_results_skipped + len(curr_task_results))
            if num_results_skipped > 0:
                logger.warning('Skipped {} out of date results ({:.2f}%)'.format(
                    num_results_skipped, 100. * frac_results_skipped))

            # Assemble results (ES)
            noise_inds_n = np.concatenate([r.noise_inds_n for r in curr_task_results])
            returns_n2 = np.concatenate([r.returns_n2 for r in curr_task_results])
            lengths_n2 = np.concatenate([r.lengths_n2 for r in curr_task_results])
            assert noise_inds_n.shape[0] == returns_n2.shape[0] == lengths_n2.shape[0]
            # Process returns (ES)
            if config.return_proc_mode == 'centered_rank':
                proc_returns_n2 = compute_centered_ranks(returns_n2)
            elif config.return_proc_mode == 'sign':
                proc_returns_n2 = np.concatenate([r.signreturns_n2 for r in curr_task_results])
            elif config.return_proc_mode == 'centered_sign_rank':
                proc_returns_n2 = compute_centered_ranks(np.concatenate([r.signreturns_n2 for r in curr_task_results]))
            else:
                raise NotImplementedError(config.return_proc_mode)
            # Compute and take step (ES)
            g, count = batched_weighted_sum(
                proc_returns_n2[:, 0] - proc_returns_n2[:, 1],
                (noise.get(idx, policy.num_params) for idx in noise_inds_n),
                batch_size=500  # why is this hard-coded?
            )
            g /= returns_n2.size
            assert g.shape == (policy.num_params,) and g.dtype == np.float32 and count == len(noise_inds_n)
            update_ratio = optimizer.update(-g + config.l2coeff * theta)

            # Update ob stat (we're never running the policy in the master, but we might be snapshotting the policy)
            if policy.needs_ob_stat:
                policy.set_ob_stat(ob_stat.mean, ob_stat.std)

            step_tend = time.time()
            tlogger.record_tabular("es/EpRewMean", returns_n2.mean())
            tlogger.record_tabular("es/EpRewStd", returns_n2.std())
            tlogger.record_tabular("es/EpLenMean", lengths_n2.mean())

            tlogger.record_tabular("es/EvalEpRewMean", np.nan if not eval_rets else np.mean(eval_rets))
            tlogger.record_tabular("es/EvalEpRewStd", np.nan if not eval_rets else np.std(eval_rets))
            tlogger.record_tabular("es/EvalEpLenMean", np.nan if not eval_rets else np.mean(eval_lens))
            tlogger.record_tabular("es/EvalPopRank", np.nan if not eval_rets else (
                np.searchsorted(np.sort(returns_n2.ravel()), eval_rets).mean() / returns_n2.size))   # ???
            tlogger.record_tabular("es/EvalEpCount", len(eval_rets))

            tlogger.record_tabular("es/Norm", float(np.square(policy.get_trainable_flat()).sum()))
            tlogger.record_tabular("es/GradNorm", float(np.square(g).sum()))
            tlogger.record_tabular("es/UpdateRatio", float(update_ratio))

            tlogger.record_tabular("es/EpisodesThisIter", lengths_n2.size)
            tlogger.record_tabular("es/EpisodesSoFar", episodes_so_far)
            tlogger.record_tabular("es/TimestepsThisIter", lengths_n2.sum())
            tlogger.record_tabular("es/TimestepsSoFar", timesteps_so_far)

            num_unique_workers = len(set(worker_ids))
            tlogger.record_tabular("es/UniqueWorkers", num_unique_workers)
            tlogger.record_tabular("es/UniqueWorkersFrac", num_unique_workers / len(worker_ids))
            tlogger.record_tabular("es/ResultsSkippedFrac", frac_results_skipped)
            tlogger.record_tabular("es/ObCount", ob_count_this_batch)

            tlogger.record_tabular("es/TimeElapsedThisIter", step_tend - step_tstart)
            tlogger.record_tabular("es/TimeElapsed", step_tend - tstart)
            tlogger.dump_tabular()
        else:  # A3C master step
            # Pop off results for the current task
            eval_rets, eval_lens, returns, lengths, worker_ids = [], [], [], [], []
            num_episodes_popped, num_updates_skipped, num_updates_popped, num_timesteps_popped = 0, 0, 0, 0
            # ob_count_this_batch = 0

            eps_per_batch = config.episodes_per_batch / 20
            tsteps_per_batch = config.timesteps_per_batch / 20
            while num_episodes_popped < eps_per_batch or num_timesteps_popped < tsteps_per_batch:
                # Wait for a result
                task_id, f = master.pop_result()
                assert isinstance(task_id, int) and isinstance(f, Fetched)
                worker_ids.append(f.worker_id)
                if f.terminal:  # full rollout
                    # additional gradient update on entire rollout?
                    # would it ever be an issue here of task != curr_task?
                    # if task_id == curr_task_id:
                    returns.append(f.ep_return)
                    lengths.append(f.ep_length)

                    # Update counts
                    num_episodes_popped += 1
                    episodes_so_far += 1
                    # timesteps_so_far += config.num_local_steps  # technically wrong
                else:
                    if f.eval_length is not None:
                        # This was an eval job
                        episodes_so_far += 1
                        timesteps_so_far += f.eval_length
                        # Store the result only for the current task
                        if task_id == curr_task_id:
                            eval_rets.append(f.eval_return)
                            eval_lens.append(f.eval_length)
                    else:
                        # Update counts
                        updates_so_far += 1
                        timesteps_so_far += config.num_local_steps  # TODO: improve on this
                        # Store results only for current tasks
                        if task_id == curr_task_id:
                            num_updates_popped += 1
                            num_timesteps_popped += config.num_local_steps
                            # Update ob stats
                            # if policy.needs_ob_stat and result.ob_count > 0:
                            #     ob_stat.increment(result.ob_sum, result.ob_sumsq, result.ob_count)
                            #     ob_count_this_batch += result.ob_count
                        else:
                            num_updates_skipped += 1

                        # gradient updates for every partial rollout?
                        g = f.grads
                        assert g.shape == (policy.num_params,) and g.dtype == np.float32
                        update_ratio = optimizer.update(-g + config.l2coeff * theta)

            # Compute skip fraction
            frac_updates_skipped = num_updates_skipped / (num_updates_skipped + num_updates_popped)
            if num_updates_skipped > 0:
                logger.warning('Skipped {} out of date results ({:.2f}%)'.format(
                    num_updates_popped, 100. * frac_updates_skipped))

            # Update ob stat (we're never running the policy in the master, but we might be snapshotting the policy)
            # if policy.needs_ob_stat:
            #     policy.set_ob_stat(ob_stat.mean, ob_stat.std)

            step_tend = time.time()
            tlogger.record_tabular("a3c/EpRewMean", np.mean(returns))
            tlogger.record_tabular("a3c/EpRewStd", np.std(returns))
            tlogger.record_tabular("a3c/EpLenMean", np.mean(lengths))

            tlogger.record_tabular("a3c/EvalEpRewMean", np.nan if not eval_rets else np.mean(eval_rets))
            tlogger.record_tabular("a3c/EvalEpRewStd", np.nan if not eval_rets else np.std(eval_rets))
            tlogger.record_tabular("a3c/EvalEpLenMean", np.nan if not eval_rets else np.mean(eval_lens))
            tlogger.record_tabular("a3c/EvalPopRank", np.nan if not eval_rets else (
                np.searchsorted(np.sort(returns_n2.ravel()), eval_rets).mean() / returns_n2.size))  # ???
            tlogger.record_tabular("a3c/EvalEpCount", len(eval_rets))

            tlogger.record_tabular("a3c/Norm", float(np.square(policy.get_trainable_flat()).sum()))
            # tlogger.record_tabular("a3c/GradNorm", float(np.square(g).sum()))
            # tlogger.record_tabular("a3c/UpdateRatio", float(update_ratio))

            tlogger.record_tabular("a3c/UpdatesThisIter", num_updates_popped)
            tlogger.record_tabular("a3c/UpdatesSoFar", updates_so_far)
            tlogger.record_tabular("a3c/TimestepsThisIter", num_timesteps_popped)
            tlogger.record_tabular("a3c/TimestepsSoFar", timesteps_so_far)

            num_unique_workers = len(set(worker_ids))
            tlogger.record_tabular("a3c/UniqueWorkers", num_unique_workers)
            tlogger.record_tabular("a3c/UniqueWorkersFrac", num_unique_workers / len(worker_ids))
            tlogger.record_tabular("a3c/UpdatesSkippedFrac", frac_updates_skipped)
            # tlogger.record_tabular("ObCount", ob_count_this_batch)

            tlogger.record_tabular("a3c/TimeElapsedThisIter", step_tend - step_tstart)
            tlogger.record_tabular("a3c/TimeElapsed", step_tend - tstart)
            tlogger.dump_tabular()

        # save a snapshot of the policy
        if config.snapshot_freq != 0 and curr_task_id % config.snapshot_freq == 0:
            import os.path as osp
            # filename = osp.join(tlogger.get_dir(), 'snapshot_iter{:05d}_rew{}.h5'.format(
            #     curr_task_id,
            #     np.nan if not eval_rets else int(np.mean(eval_rets))
            # ))
            filename = osp.join(tlogger.get_dir(), 'snapshot_iter{:05d}.h5'.format(curr_task_id))
            policy.save(filename)
            tlogger.log('Saved snapshot {}'.format(filename))

        if curr_task_id % config.switch_freq == 0:
            # want exponentially decreasing probability? make this a hyperparameter if so?
            es_a3c = rs.rand() < np.power(config.es_a3c_prob, curr_task_id)


def rollout_and_update_ob_stat(policy, env, rs, task_ob_stat, calc_obstat_prob):
    if policy.needs_ob_stat and calc_obstat_prob != 0 and rs.rand() < calc_obstat_prob:   # why a probability?
        rollout_rews, rollout_len, obs = policy.rollout(env, save_obs=True, random_stream=rs)
        task_ob_stat.increment(obs.sum(axis=0), np.square(obs).sum(axis=0), len(obs))
    else:
        rollout_rews, rollout_len = policy.rollout(env, random_stream=rs)
    return rollout_rews, rollout_len


def run_worker(relay_redis_cfg, noise, min_task_runtime=1.):  # what should min_task_runtime default be?
    logger.info('run_worker: {}'.format(locals()))
    assert isinstance(noise, SharedNoiseTable)
    worker = WorkerClient(relay_redis_cfg)
    exp = worker.get_experiment()
    config, env, sess, policy = setup(exp, single_threaded=True)
    rs = np.random.RandomState()
    worker_id = rs.randint(2 ** 31)   # don't have to order worker_ids and guarantee no two same id?
    # summary_writer = tf.summary.FileWriter(log_dir + "_%d" % worker_id)

    assert policy.needs_ob_stat == (config.calc_obstat_prob != 0)

    # TODO: set up local and global a3c steps
    ac = tf.placeholder(tf.float32, [None, env.action_space.n], name='ac')
    adv = tf.placeholder(tf.float32, [None], name='adv')
    r = tf.placeholder(tf.float32, [None], name='r')

    log_prob_tf = tf.nn.log_softmax(policy.logits)
    prob_tf = tf.nn.softmax(policy.logits)

    # the "policy gradients" loss: its derivative is precisely the policy gradient
    # notice that ac is a placeholder that is provided externally.
    # adv will contain the advantages, as calculated in process_rollout
    pi_loss = - tf.reduce_sum(tf.reduce_sum(log_prob_tf * ac, [1]) * adv)

    vf_loss = 0.5 * tf.reduce_sum(tf.square(policy.vf - r))  # loss of value function
    vf_loss = tf.cast(vf_loss, tf.float32)
    entropy = - tf.reduce_sum(prob_tf * log_prob_tf)

    bs = tf.to_float(tf.shape(policy.x)[0])  # batch size
    loss = pi_loss + 0.5 * vf_loss - entropy * 0.01

    while True:
        # retrieve and check current task
        task_id, task_data = worker.get_current_task()
        task_tstart = time.time()
        assert isinstance(task_id, int) and isinstance(task_data, Task)

        if not task_data.a3c:  # ES worker step
            # do ob_stat stuff for a3c as well?
            if policy.needs_ob_stat:
                policy.set_ob_stat(task_data.ob_mean, task_data.ob_std)

            if rs.rand() < config.eval_prob:
                # Evaluation: noiseless weights and noiseless actions
                policy.set_trainable_flat(task_data.params)
                eval_rews, eval_length = policy.rollout(env)
                eval_return = eval_rews.sum()
                logger.info('Eval rewards: {}'.format(eval_rews))
                logger.info('Eval result: task={} return={:3f} length={}'.format(task_id, eval_return, eval_length))
                worker.push_result(task_id, Result(
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
                task_ob_stat = RunningStat(env.observation_space.shape, eps=0.)   # eps=0 bc only incrementing

                while not noise_inds or time.time() - task_tstart < min_task_runtime:
                    noise_idx = noise.sample_index(rs, policy.num_params)   # sampling indices for random noise values?
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

                worker.push_result(task_id, Result(
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
        else:  # A3C worker step
            if rs.rand() < config.eval_prob * 10:
                # Evaluation: noiseless weights and noiseless actions
                policy.set_trainable_flat(task_data.params)
                eval_rews, eval_length = policy.rollout(env)
                eval_return = eval_rews.sum()
                logger.info('Eval rewards: {}'.format(eval_rews))
                logger.info('Eval result: task={} return={:3f} length={}'.format(task_id, eval_return, eval_length))
                worker.push_result(task_id, Fetched(
                    worker_id=worker_id,
                    grads=None,
                    terminal=None,
                    ep_return=None,
                    ep_length=None,
                    eval_return=eval_return,
                    eval_length=eval_length
                ))
            else:
                # TODO: should also do ob_stat update with A3C?
                policy.set_trainable_flat(task_data.params)

                # matters not inputting master variables?
                # too large to communicate without clipping
                grads = tf.gradients(loss, policy.trainable_variables)
                grads, _ = tf.clip_by_global_norm(grads, 40.0)  # why 40.0?
                grads = tf.concat([tf.reshape(grad, [U.numel(v)]) for (v, grad)
                                   in zip(policy.trainable_variables, grads)], 0)  # flatten gradients

                # tf.summary.scalar("model/policy_loss", pi_loss / bs)
                # tf.summary.scalar("model/value_loss", vf_loss / bs)
                # tf.summary.scalar("model/entropy", entropy / bs)
                # tf.summary.scalar("model/loss", loss / bs)
                # tf.summary.image("model/state", policy.x)
                # tf.summary.scalar("model/grad_global_norm", tf.global_norm(grads))
                # tf.summary.scalar("model/var_global_norm", tf.global_norm(policy.trainable_variables))

                rollout_provider = policy.env_runner(env, num_local_steps=config.num_local_steps)
                rollout = p_rollout = next(rollout_provider)

                # tf.summary.scalar("max_value", tf.reduce_max(policy.vf))
                # tf.summary.scalar("min_value", tf.reduce_min(policy.vf))
                # tf.summary.scalar("mean_value", tf.reduce_mean(policy.vf))
                # tf.summary.scalar("reward_max", tf.reduce_max(rollout.rewards))
                # tf.summary.scalar("reward_min", tf.reduce_min(rollout.rewards))
                # tf.summary.scalar("reward_mean", tf.reduce_mean(rollout.rewards))
                # tf.summary.histogram("reward_targets", rollout.rewards)
                # tf.summary.histogram("values", policy.vf)
                # summary_op = tf.summary.merge_all()

                while not rollout.terminal:  # TODO: incorporate local and global steps
                    # TODO: local and global steps per batch?
                    # TODO: set gamma and lambda_ as hyperparameters
                    batch = process_rollout(p_rollout, gamma=0.99, lambda_=1.0)

                    # should_compute_summary =
                    # if should_compute_summary:
                    #     fetches = [summary_op, grads, global_step]
                    # else:
                    #     fetches = [grads, global_step]

                    feed_dict = {
                        policy.x: batch.si,
                        ac: batch.a,
                        adv: batch.adv,
                        r: batch.r,
                        policy.state_in[0]: batch.features[0],
                        policy.state_in[1]: batch.features[1]
                    }

                    # summ, fetched = sess.run([summary_op, grads], feed_dict=feed_dict)
                    gradients = sess.run(grads, feed_dict=feed_dict)
                    # if should_compute_summary:
                    # summary_writer.add_summary(tf.Summary.FromString(fetched[0]), fetched[-1])
                    # summary_writer.add_summary(tf.Summary.FromString(summ))
                    # summary_writer.flush()

                    worker.push_result(task_id, Fetched(
                        worker_id=worker_id,
                        grads=gradients,
                        terminal=rollout.terminal,
                        ep_return=None,
                        ep_length=None,
                        eval_return=None,
                        eval_length=None
                    ))

                    p_rollout = next(rollout_provider)
                    rollout.extend(p_rollout)

                # full rollout
                worker.push_result(task_id, Fetched(
                    worker_id=worker_id,
                    grads=None,
                    terminal=rollout.terminal,
                    ep_return=np.sum(rollout.rewards),
                    ep_length=len(rollout.rewards),
                    eval_return=None,
                    eval_length=None
                ))
