# Modified from policies.py from OpenAI's evolutionary-strategies-starter project
import logging

try:
    import cPickle as pickle
except ImportError:
    import pickle

import h5py   # TODO: study HDF5 / h5py
import numpy as np
import tensorflow as tf

from es_distributed import tf_util as U

logger = logging.getLogger(__name__)

#TODO: implement other Env classes


class Policy:
    def __init__(self, *args, **kwargs):
        self.args, self.kwargs = args, kwargs
        self.scope = self._initialize(*args, **kwargs)   # scope?
        self.all_variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, self.scope.name)

        self.trainable_variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.scope.name)
        self.num_params = sum(int(np.prod(v.get_shape().as_list())) for v in self.trainable_variables)
        self._setfromflat = U.SetFromFlat(self.trainable_variables)
        self._getflat = U.GetFlat(self.trainable_variables)

        # simply for logging/debugging purposes?
        logger.info('Trainable variables ({} parameters)'.format(self.num_params))
        for v in self.trainable_variables:
            shp = v.get_shape().as_list()
            logger.info('- {} shape:{} size:{}'.format(v.name, shp, np.prod(shp)))
        logger.info('All variables')
        for v in self.all_variables:
            shp = v.get_shape().as_list()
            logger.info('- {} shape:{} size:{}'.format(v.name, shp, np.prod(shp)))

        placeholders = [tf.placeholder(v.value().dtype, v.get_shape().as_list()) for v in self.all_variables]
        self.set_all_vars = U.function(
            inputs=placeholders,
            outputs=[],
            updates=[tf.group(*[v.assign(p) for v, p in zip(self.all_variables, placeholders)])]
        )

    def _initialize(self, *args, **kwargs):
        raise NotImplementedError

    def save(self, filename):
        assert filename.endswith('.h5')
        with h5py.File(filename, 'w') as f:
            for v in self.all_variables:
                f[v.name] = v.eval()   # evaluating the tf variable?
            f.attrs['name'] = type(self).__name__
            f.attrs['args_and_kwargs'] = np.void(pickle.dumps((self.args, self.kwargs), protocol=-1))   # ???

    @classmethod
    def Load(cls, filename, extra_kwargs=None):
        with h5py.File(filename, 'r') as f:
            args, kwargs = pickle.loads(f.attrs['args_and_kwargs'].tostring())
            if extra_kwargs:
                kwargs.update(extra_kwargs)
            policy = cls(*args, **kwargs)
            policy.set_all_vars(*[f[v.name][...] for v in policy.all_variables])
        return policy

    # === Rollouts/training ===

    def rollout(self, env, render=False, save_obs=False, random_stream=None):
        """
        If random_stream is provided, the rollout will take noisy actions with noise drawn from that stream.
        Otherwise, no action noise will be added.
        """
        rews = []
        t = 0
        if save_obs:
            obs = []
        ob = env.reset()
        while True:
            ac = self.act(np.squeeze(ob[None]), random_stream=random_stream)[0]
            if save_obs:
                obs.append(ob)
            ob, rew, done, _ = env.step(ac.argmax())   # want the argmax?
            rews.append(rew)
            t += 1
            if render:
                env.render()
            if done:
                break
        rews = np.array(rews, dtype=np.float32)
        if save_obs:
            return rews, t, np.array(obs)
        return rews, t

    def act(self, ob, random_stream=None):
        raise NotImplementedError

    def set_trainable_flat(self, x):
        self._setfromflat(x)

    def get_trainable_flat(self):
        return self._getflat()

    @property
    def needs_ob_stat(self):
        raise NotImplementedError

    def set_ob_stat(self, ob_mean, ob_std):
        raise NotImplementedError


# for continuous action spaces
# def bins(x, dim, num_bins, name):
#     scores = U.dense(x, dim * num_bins, name, U.normc_initializer(0.01))
#     scores_nab = tf.reshape(scores, [-1, dim, num_bins])
#     return tf.argmax(scores_nab, 2)   # 0 ... num_bins-1


class GoPolicy(Policy):
    # ob_space = Box(3, 19, 19)
    # ac_space = Discrete(363)   19*19 + 2
    def _initialize(self, ob_space, ac_space, ac_noise_std, nonlin_type, hidden_dims, connection_type):
        self.ac_space = ac_space
        self.ac_noise_std = ac_noise_std
        self.hidden_dims = hidden_dims
        self.connection_type = connection_type

        # what is [nonlin_type] doing here? also wtf is an elu?
        self.nonlin = {'tanh': tf.tanh, 'relu': tf.nn.relu, 'lrelu': U.lrelu, 'elu': tf.nn.elu}[nonlin_type]

        with tf.variable_scope(type(self).__name__) as scope:
            # Observation normalization <-- is this necessary for Go?
            ob_mean = tf.get_variable(
                'ob_mean', ob_space.shape, tf.float32, tf.constant_initializer(np.nan), trainable=False)
            ob_std = tf.get_variable(
                'ob_std', ob_space.shape, tf.float32, tf.constant_initializer(np.nan), trainable=False)
            # what are in_mean and in_std? batch-specific statistics?
            in_mean = tf.placeholder(tf.float32, ob_space.shape)
            in_std = tf.placeholder(tf.float32, ob_space.shape)
            self._set_ob_mean_std = U.function([in_mean, in_std], [], updates=[
                tf.assign(ob_mean, in_mean),
                tf.assign(ob_std, in_std),
            ])

            # Policy network
            o = tf.placeholder(tf.float32, list(ob_space.shape))   # o = input placeholder variable?
            a = self._make_net(tf.clip_by_value((o - ob_mean) / ob_std, -5.0, 5.0))
            self._act = U.function([o], a)   # pass o to nn a?
        return scope

    def _make_net(self, o):
        # Process observation
        if self.connection_type == 'ff':
            x = o
            for ilayer, hd in enumerate(self.hidden_dims):
                x = self.nonlin(U.dense(x, hd, 'l{}'.format(ilayer), U.normc_initializer(1.0)))
        else:
            raise NotImplementedError(self.connection_type)

        # Map to action
        adim = self.ac_space.n
        a = U.dense(x, adim, 'out', U.normc_initializer(0.01))
        return a

    def act(self, ob, random_stream=None):
        a = self._act(ob)
        if random_stream is not None and self.ac_noise_std != 0:
            a += random_stream.randn(*a.shape) * self.ac_noise_std
        return a   # softmax vector

    @property
    def needs_ob_stat(self):   # necessary for GoEnv?
        return True

    @property
    def needs_ref_batch(self):   # necessary for GoEnv?
        return False

    def set_ob_stat(self, ob_mean, ob_std):
        self._set_ob_mean_std(ob_mean, ob_std)

    #def initialize_from(self, filename, ob_stat=None):