# Modified from policies.py from OpenAI's evolutionary-strategies-starter project
import logging
from collections import namedtuple

try:
    import cPickle as pickle
except ImportError:
    import pickle

import h5py   # TODO: study HDF5 / h5py
import numpy as np
import tensorflow as tf
import tensorflow.contrib.rnn as rnn

from es_distributed import tf_util as U

logging.basicConfig(filename='experiment.log', level=logging.INFO)
logger = logging.getLogger(__name__)


class Policy:
    def __init__(self, *args, **kwargs):
        self.args, self.kwargs = args, kwargs
        self.scope = self._initialize(*args, **kwargs)   # scope?
        self.all_variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, self.scope.name)

        self.trainable_variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.scope.name)
        self.num_params = sum(int(np.prod(v.get_shape().as_list())) for v in self.trainable_variables)
        self._setfromflat = U.SetFromFlat(self.trainable_variables)
        self._getflat = U.GetFlat(self.trainable_variables)

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
            # TODO: it would be nice to avoid pickle, but it's convenient to pass Python objects to _initialize
            # (like Gym spaces or numpy arrays)
            f.attrs['name'] = type(self).__name__
            #f.attrs['args_and_kwargs'] = np.void(pickle.dumps((self.args, self.kwargs), protocol=-1))   # ???

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
        logger.info('Rewards: {}'.format(rews))
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
    def _initialize(self, ob_space, ac_space, ac_noise_std, nonlin_type, hidden_dims, lstm_size):
        self.ac_space = ac_space
        self.ac_noise_std = ac_noise_std
        self.hidden_dims = hidden_dims
        self.lstm_size = lstm_size

        self.nonlin = {'tanh': tf.tanh, 'relu': tf.nn.relu, 'lrelu': U.lrelu, 'elu': tf.nn.elu}[nonlin_type]

        with tf.variable_scope(type(self).__name__) as scope:
            # Observation normalization
            ob_mean = tf.get_variable(
                'ob_mean', ob_space.shape, tf.float32, tf.constant_initializer(np.nan), trainable=False)
            ob_std = tf.get_variable(
                'ob_std', ob_space.shape, tf.float32, tf.constant_initializer(np.nan), trainable=False)
            in_mean = tf.placeholder(tf.float32, ob_space.shape)
            in_std = tf.placeholder(tf.float32, ob_space.shape)
            self._set_ob_mean_std = U.function([in_mean, in_std], [], updates=[
                tf.assign(ob_mean, in_mean),
                tf.assign(ob_std, in_std),
            ])

            # Policy network
            # would below work for go?
            self.x = x = tf.placeholder(tf.float32, [None] + list(ob_space.shape))
            for ilayer, hd in enumerate(self.hidden_dims):
                # have filter size = board size (19 x 19)?
                # decrease dimensionality or nah?
                x = self.nonlin(U.conv2d(x, hd, "l{}".format(ilayer), [3, 3], [1, 1]))
            # introduce a "fake" batch dimension of 1 after flatten so that we can do LSTM over time dim
            x = tf.expand_dims(U.flatten(x), [0])

            lstm = rnn.BasicLSTMCell(self.lstm_size, state_is_tuple=True)
            step_size = tf.shape(self.x)[:1]

            c_init = np.zeros((1, lstm.state_size.c), np.float32)
            h_init = np.zeros((1, lstm.state_size.h), np.float32)
            self.state_init = [c_init, h_init]
            c_in = tf.placeholder(tf.float32, [1, lstm.state_size.c])
            h_in = tf.placeholder(tf.float32, [1, lstm.state_size.h])
            self.state_in = [c_in, h_in]

            state_in = rnn.LSTMStateTuple(c_in, h_in)
            lstm_outputs, lstm_state = tf.nn.dynamic_rnn(
                lstm, x, initial_state=state_in, sequence_length=step_size,
                time_major=False)
            lstm_c, lstm_h = lstm_state
            x = tf.reshape(lstm_outputs, [-1, self.lstm_size])
            self.logits = U.dense(x, self.ac_space.n, 'action', U.normc_initializer(0.01))
            self.state_out = [lstm_c[:1, :], lstm_h[:1, :]]
            self.sample = U.categorical_sample(self.logits, self.ac_space.n)[0, :]

        return scope

    def get_initial_features(self):
        return self.state_init

    def act(self, ob, c0, h0, random_stream=None):
        sess = U.get_session()
        a, c1, h1 = sess.run([self.sample] + self.state_out,
                             {self.x: [ob], self.state_in[0]: c0, self.state_in[1]: h0})
        if random_stream is not None and self.ac_noise_std != 0:
            a += random_stream.randn(*a.shape) * self.ac_noise_std
        return a, c1, h1

    def rollout(self, env, render=False, save_obs=False, random_stream=None):
        """
        If random_stream is provided, the rollout will take noisy actions with noise drawn from that stream.
        Otherwise, no action noise will be added.
        """
        rews = []
        t = 0
        if save_obs:
            obs = []
        last_ob = env.reset()
        last_features = self.get_initial_features()
        while True:
            fetched = self.act(last_ob, *last_features, random_stream=random_stream)
            ac, last_features = fetched[0], fetched[1:]
            if save_obs:
                obs.append(last_ob)
            #ac = ac.argmax()
            ac = np.random.choice(np.arange(len(ac)), p=ac)  # is it a softmax vector?
            last_ob, rew, done, _ = env.step(ac.argmax())
            rews.append(rew)
            t += 1
            if render:
                env.render()
            if np.abs(rew) == 1:  # helps avoid weird reward 0 bug
                break
        rews = np.array(rews, dtype=np.float32)
        if save_obs:
            return rews, t, np.array(obs)
        return rews, t

    @property
    def needs_ob_stat(self):   # necessary for GoEnv?
        return True

    @property
    def needs_ref_batch(self):   # necessary for GoEnv?
        return False

    def set_ob_stat(self, ob_mean, ob_std):
        self._set_ob_mean_std(ob_mean, ob_std)

    #def initialize_from(self, filename, ob_stat=None):


# TODO: make consistent with es code
class PartialRollout(object):
    """
    a piece of a complete rollout. We run our agent, and process its experience
    once it has processed enough steps. Only for A3C.
    """
    def __init__(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.values = []
        self.r = 0.0
        self.terminal = False
        self.features = []

    def add(self, state, action, reward, value, terminal, features):
        self.states += [state]
        self.actions += [action]
        self.rewards += [reward]
        self.values += [value]
        self.terminal = terminal
        self.features += [features]

    def extend(self, other):
        assert not self.terminal
        self.states.extend(other.states)
        self.actions.extend(other.actions)
        self.rewards.extend(other.rewards)
        self.values.extend(other.values)
        self.r = other.r
        self.terminal = other.terminal
        self.features.extend(other.features)


# so far, only tested on Pong and Breakout (for which the ob_space and ac_space are the same structure)
# from Britz: limit action space for Pong and Breakout?
class AtariPolicy(Policy):
    # ob_space = Box(210, 160, 3)  # pixels most likely
    # ac_space = Discrete(6)
    def _initialize(self, ob_space, ac_space, ac_noise_std, nonlin_type, hidden_dims, lstm_size):
        self.ac_space = ac_space
        self.ac_noise_std = ac_noise_std
        self.hidden_dims = hidden_dims
        self.lstm_size = lstm_size

        self.nonlin = {'tanh': tf.tanh, 'relu': tf.nn.relu, 'lrelu': U.lrelu, 'elu': tf.nn.elu}[nonlin_type]

        with tf.variable_scope(type(self).__name__) as scope:
            # Observation normalization  <-- where is the actual normalization occurring?
            ob_mean = tf.get_variable(
                'ob_mean', ob_space.shape, tf.float32, tf.constant_initializer(np.nan), trainable=False)
            ob_std = tf.get_variable(
                'ob_std', ob_space.shape, tf.float32, tf.constant_initializer(np.nan), trainable=False)
            in_mean = tf.placeholder(tf.float32, ob_space.shape)
            in_std = tf.placeholder(tf.float32, ob_space.shape)
            self._set_ob_mean_std = U.function([in_mean, in_std], [], updates=[
                tf.assign(ob_mean, in_mean),
                tf.assign(ob_std, in_std),
            ])

            # Policy network
            self.x = x = tf.placeholder(tf.float32, [None] + list(ob_space.shape))
            for ilayer, hd in enumerate(self.hidden_dims):
                x = self.nonlin(U.conv2d(x, hd, 'l{}'.format(ilayer), [3, 3], [2, 2]))
            # introduce a "fake" batch dimension of 1 after flatten so that we can do LSTM over time dim
            x = tf.expand_dims(U.flatten(x), [0])

            lstm = rnn.BasicLSTMCell(self.lstm_size, state_is_tuple=True)
            self.state_size = lstm.state_size
            step_size = tf.shape(self.x)[:1]

            c_init = np.zeros((1, lstm.state_size.c), np.float32)
            h_init = np.zeros((1, lstm.state_size.h), np.float32)
            self.state_init = [c_init, h_init]
            c_in = tf.placeholder(tf.float32, [1, lstm.state_size.c])
            h_in = tf.placeholder(tf.float32, [1, lstm.state_size.h])
            self.state_in = [c_in, h_in]

            state_in = rnn.LSTMStateTuple(c_in, h_in)
            lstm_outputs, lstm_state = tf.nn.dynamic_rnn(
                lstm, x, initial_state=state_in, sequence_length=step_size,
                time_major=False)
            lstm_c, lstm_h = lstm_state
            x = tf.reshape(lstm_outputs, [-1, self.lstm_size])
            self.logits = U.dense(x, len(self.ac_space), 'action', U.normc_initializer(0.01))
            self.ac_probs = tf.nn.softmax(self.logits)
            self.vf = tf.reshape(U.dense(x, 1, 'value', U.normc_initializer(1.0)), [-1])  # only for A3C
            self.state_out = [lstm_c[:1, :], lstm_h[:1, :]]
            # self.sample = U.categorical_sample(self.logits, self.ac_space.n)[0, :]

        return scope

    def get_initial_features(self):
        return self.state_init

    def act(self, ob, c0, h0, random_stream=None):
        sess = U.get_session()
        a, c1, h1 = sess.run([self.ac_probs, self.vf] + self.state_out,
                             {self.x: [ob], self.state_in[0]: c0, self.state_in[1]: h0})
        # if random_stream is not None and self.ac_noise_std != 0:  # only use if using one-hot and argmax
        #     a += random_stream.randn(*a.shape) * self.ac_noise_std
        return a, c1, h1   # softmax vector?

    def value(self, ob, c, h):
        sess = U.get_session()
        return sess.run(self.vf, {self.x: [ob], self.state_in[0]: c, self.state_in[1]: h})[0]

    def rollout(self, env, render=False, save_obs=False, random_stream=None):  # for ES
        """
        If random_stream is provided, the rollout will take noisy actions with noise drawn from that stream.
        Otherwise, no action noise will be added.
        """
        rews = []
        t = 0
        if save_obs:
            obs = []
        last_ob = env.reset()
        last_features = self.get_initial_features()
        while True:
            fetched = self.act(last_ob, *last_features, random_stream=random_stream)
            ac, last_features = fetched[0], fetched[2:]
            ac = ac.squeeze()
            if save_obs:
                obs.append(last_ob)
            # ac = ac.argmax()  # want argmax when using one-hot and random_stream
            ac = np.random.choice(np.arange(len(ac)), p=ac)  # when using action probs
            last_ob, rew, done, _ = env.step(ac)
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

    # TODO: make consistent with es code
    def partial_rollout(self, env, num_local_steps, render=False, random_stream=False):
        # summary writer? save_obs?
        last_state = env.reset()
        last_features = self.get_initial_features()
        length = 0
        rewards = 0

        while True:
            terminal_end = False
            p_rollout = PartialRollout()

            for _ in range(num_local_steps):
                fetched = self.act(last_state, *last_features, random_stream=random_stream)
                action, value_, features = fetched[0], fetched[1], fetched[2:]
                action = action.squeeze()
                # action = action.argmax()  # only want the argmax when using one-hot and random_stream
                action = np.random.choice(np.arange(len(action)), p=action)  # when using action probs
                state, reward, terminal, info = env.step(action)
                if render:
                    env.render()

                # collect the experience
                p_rollout.add(last_state, action, reward, value_, terminal, last_features)
                length += 1
                rewards += reward

                last_state = state
                last_features = features

                # if info:  # for tf summary writing, if want to implement

                if terminal:  # want to implement timestep_limit? will env always have?
                    terminal_end = True
                    last_features = self.get_initial_features()
                    length = 0
                    rewards = 0
                    break

            if not terminal_end:
                p_rollout.r = self.value(last_state, *last_features)

            # once we have enough experience, yield it
            yield p_rollout

    @property
    def needs_ob_stat(self):   # necessary?
        return True

    @property
    def needs_ref_batch(self):   # necessary?
        return False

    def set_ob_stat(self, ob_mean, ob_std):
        self._set_ob_mean_std(ob_mean, ob_std)

    #def initialize_from(self, filename, ob_stat=None):