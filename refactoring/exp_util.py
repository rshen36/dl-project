# specific utility code for master, worker, and relays
import logging
import numpy as np
from collections import namedtuple

logger = logging.getLogger(__name__)


Config = namedtuple('Config', [
    'l2coeff', 'noise_stdev', 'episodes_per_batch',
    'calc_obstat_prob', 'eval_prob', 'snapshot_freq',
    'return_proc_mode', 'episode_cutoff_mode'
])
Task = namedtuple('Task', ['params', 'ob_mean', 'ob_std'])
Result = namedtuple('Result', [
    'worker_id',
    'noise_inds_n', 'returns_n2', 'signreturns_n2', 'lengths_n2',
    'eval_return', 'eval_length',
    'ob_sum', 'ob_sumsq', 'ob_count'
])


class RunningStat(object):
    def __init__(self, shape, eps):
        self.sum = np.zeros(shape, dtype=np.float32)
        self.sumsq = np.full(shape, eps, dtype=np.float32)   # returns array of given shape with given fill vals
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
        import ctypes, multiprocessing   # ???
        seed = 123

        # may need to adapt this number
        #count = 250000000   # 1 gigabyte of 32-bit numbers. Will actually sample 2 gigabytes below.
        count = 2500000

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


def make_session(single_threaded):
    import tensorflow as tf
    if not single_threaded: return tf.InteractiveSession()
    return tf.InteractiveSession(config=tf.ConfigProto(inter_op_parallelism_threads=1,
                                                       intra_op_parallelism_threads=1))


def itergroups(items, group_size):   # grab group of size group_size from start of items?
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


def setup(exp, single_threaded):
    import gym
    gym.undo_logger_setup()   # to allow for own logging
    from . import tf_util, policies

    config = Config(**exp['config'])
    env = gym.make(exp['env_id'])
    sess = make_session(single_threaded=single_threaded)
    policy = getattr(policies, exp['policy']['type'])(env.observation_space,  # ???
                                                      env.action_space, **exp['policy']['args'])
    tf_util.initialize()

    return config, env, sess, policy