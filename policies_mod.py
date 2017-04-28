# Modified from policies.py from OpenAI's evolutionary-strategies-starter project
import logging

try:
    import cPickle as pickle
except:
    import pickle

import h5py   # TODO: study HDF5 / h5py
import numpy as np
import tensorflow as tf

#from . import tf_util as U

logger = logging.getLogger(__name__)


class Policy:
    def __init__(self, *args, **kwargs):   # args and kwargs?
        self.args, self.kwargs = args, kwargs
        self.scope = self._initialize(*args, **kwargs)   # scope and initialize?
        self.all_variables = tf.get_collection(tf.GraphKeys.VARIABLES, self.scope.name)   # ???

        self.trainable_variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.scope.name)   # ???
        self.num_params = sum(int(np.prod(v.get_shape().as_list())) for v in self.trainable_variables)
        #self._setfromflat = U.SetFromFlat(self.trainable_variables)   # ???
        #self._getflat = U.GetFlat(self.trainable_variables)   # ???

        # simply for logging/debugging purposes?
        logger.info('Trainable variables ({} parameters)'.format(self.num_params))
        for v in self.trainable_variables:
            shp = v.get_shape().as_list()
            logger.info('- {} shape:{} size:{}'.format(v.name, shp, np.prod(shp)))
        logger.info('All variables')
        for v in self.all_variables:
            shp = v.get_shape().as_list()
            logger.info('- {} shape:{} size:{}'.format(v.name, shp, np.prod(shp)))

        placeholders = [tf.placeholders(v.value().dtype, v.get_shape().as_list()) for v in self.all_variables]
        #self.set_all_vars = U.function()

    def _initialize(self, *args, **kwargs):
        raise NotImplementedError

    def save(self, filename):
        assert filename.endswith('.h5')
        with h5py.File(filename, 'w') as f:
            for v in self.all_variables:
                f[v.name] = v.eval()   # ???
            # TODO: it would be nice to avoid pickle, but it's convenient to pass Python objects to _initialize
            # (like Gym spaces or numpy arrays)
            f.attrs['name'] = type(self).__name__
            f.attrs['args_and_kwargs'] = np.void(pickle.dumps((self.args, self.kwargs), protocol=-1))   # ???

    @classmethod   # TODO: study up on decorators
    def Load(cls, filename, extra_kwargs=None):
        with h5py.File(filename, 'r') as f:
            args, kwargs = pickle.loads(f.attrs['args_and_kwargs'].tostring())
            if extra_kwargs:
                kwargs.update(extra_kwargs)   # ???
            policy = cls(*args, **kwargs)   # ???
            policy.set_all_vars(*[f[v.name][...] for v in policy.all_variables])   # ???
        return policy

    # === Rollouts/training === <-- wut

    # note: '*' argument is to force the passing of named arguments
    def rollout(self, env, *, render=False, timestep_limit=None, save_obs=False, random_stream=None):
        """
        If random_stream is provided, the rollout will take noisy actions with noise drawn from that stream.
        Otherwise, no action noise will be added.
        """
        env_timestep_limit = env.spec.tags.get('wrapper_config.TimeLimit.max_episode_steps')   # ???
        timestep_limit = env_timestep_limit if timestep_limit is None else min(timestep_limit, env_timestep_limit)
        rews = []
        t = 0
        if save_obs:
            obs = []
        ob = env.reset()
        for _ in range(timestep_limit):
            ac = self.act(ob[None], random_stream=random_stream)[0]
            if save_obs:
                obs.append(ob)
            ob, rew, done, _ = env.step(ac)
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

    def set_trainable_flat(self, x):   # ???
        self._setfromflat(x)

    def get_trainable_flat(self):   # ???
        return self._getflat()

    @property
    def needs_ob_stat(self):
        raise NotImplementedError

    def set_ob_stat(self, ob_mean, ob_std):
        raise NotImplementedError


#def bins(x, dim, num_bins, name):



#class GoPolicy(Policy):


#class ChessPolicy(Policy):


#class MKartPolicy(Policy):


#class TorcsPolicy(Policy):


#class DDrivePolicy(Policy):

#class ...