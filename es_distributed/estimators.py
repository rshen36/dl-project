import numpy as np
import tensorflow as tf
import tensorflow.contrib.rnn as rnn

from es_distributed import tf_util as U


# TODO: test this
# TODO: replace tf summaries with tlogger
def build_shared_network(X, nonlin_type, hidden_dims, lstm_size, add_summaries=False):
    """
    Builds a CNN-LSTM network of the same architecture as used by the ES algorithm.
    This architecture was adapted from OpenAI's universe-starter-agent project.
    This network is shared by both the policy and value net.

    Args:
        X: Inputs
        nonlin_type: nonlinearity type to use
        hidden_dims: list of the dimensions of the hidden layers
        lstm_size: size of the LSTM layer
        add_summaries: If true, add layer summaries to Tensorboard. <-- keep or get rid of this?

    Returns:
        Final layer activations.  <-- I think?
    """
    out = X
    nonlin = {'tanh': tf.tanh, 'relu': tf.nn.relu, 'lrelu': U.lrelu, 'elu': tf.nn.elu}[nonlin_type]

    # add convolutional layers
    for ilayer, hd in enumerate(hidden_dims):
        out = nonlin(U.conv2d(out, hd, "l{}".format(ilayer), [3, 3], [2, 2]))
        if add_summaries:
            tf.contrib.layers.summarize_activation(out)
    out = tf.expand_dims(U.flatten(out), [0])  # does this work for this construction?

    # add LSTM layer
    lstm = rnn.BasicLSTMCell(lstm_size, state_is_tuple=True)
    step_size = tf.shape(X)[:1]  # may be wrong dimensions

    # c_init = np.zeros((1, lstm.state_size.c), np.float32)
    # h_init = np.zeros((1, lstm.state_size.h), np.float32)
    # self_state_init = [c_init, h_init]
    c_in = tf.placeholder(tf.float32, [1, lstm.state_size.c])
    h_in = tf.placeholder(tf.float32, [1, lstm.state_size.h])
    # self_state_in = [c_in, h_in]

    state_in = rnn.LSTMStateTuple(c_in, h_in)
    lstm_outputs, lstm_state = tf.nn.dynamic_rnn(
        lstm, out, initial_state=state_in, sequence_length=step_size,
        time_major=False)
    # lstm_c, lstm_h = lstm_state
    out = tf.reshape(lstm_outputs, [-1, lstm_size])
    # don't want logits layer at this point
    # self_state_out = [lstm_c[:1, :], lstm_h[:1, :]]
    if add_summaries:  # ???
        tf.contrib.layers.summarize_activation(lstm_state)
        tf.contrib.layers.summarize_activation(lstm_outputs)

    return out


class PolicyEstimator:
    """
    Policy Function approximator. Given a observation, returns probabilities
    over all possible actions.

    Args:
        num_outputs: Size of the action space.
        reuse: If true, an existing shared network will be re-used.
        trainable: If true we add train ops to the network.
            Actor threads that don't update their local models and don't need
            train ops would set this to false.
    """

    def __init__(self, num_outputs, reuse=False, trainable=True):
        self.num_outputs = num_outputs

        # Placeholders for our input
        # Our input are 4 RGB frames of shape 160, 160 each  <-- always true?
        self.states = tf.placeholder(shape=[None, 84, 84, 4], dtype=tf.uint8, name="X")
        # The TD target value
        self.targets = tf.placeholder(shape=[None], dtype=tf.float32, name="y")
        # Integer id of which action was selected
        self.actions = tf.placeholder(shape=[None], dtype=tf.int32, name="actions")

        # Normalize
        X = tf.to_float(self.states) / 255.0
        batch_size = tf.shape(self.states)[0]

        # Graph shared with Value Net
        with tf.variable_scope("shared", reuse=reuse):
            out = build_shared_network(X, add_summaries=(not reuse))

        with tf.variable_scope("policy_net"):
            self.logits = U.dense(out, num_outputs, "action", U.normc_initializer(0.01))  # ???
            self.probs = tf.nn.softmax(self.logits) + 1e-8

            self.predictions = {
                "logits": self.logits,
                "probs": self.probs
            }

            # We add entropy to the loss to encourage exploration
            self.entropy = -tf.reduce_sum(self.probs * tf.log(self.probs), 1, name="entropy")
            self.entropy_mean = tf.reduce_mean(self.entropy, name="entropy_mean")

            # Get the predictions for the chosen actions only
            gather_indices = tf.range(batch_size) * tf.shape(self.probs)[1] + self.actions
            self.picked_action_probs = tf.gather(tf.reshape(self.probs, [-1]), gather_indices)

            self.losses = - (tf.log(self.picked_action_probs) * self.targets + 0.01 * self.entropy)
            self.loss = tf.reduce_sum(self.losses, name="loss")

            # TODO: replace this with tlogger
            tf.summary.scalar(self.loss.op.name, self.loss)
            tf.summary.scalar(self.entropy_mean.op.name, self.entropy_mean)
            tf.summary.histogram(self.entropy.op.name, self.entropy)

            if trainable:
                # TODO: use optimizer specified in Config
                # self.optimizer = tf.train.AdamOptimizer(1e-4)
                self.optimizer = tf.train.RMSPropOptimizer(0.00025, 0.99, 0.0, 1e-6)
                self.grads_and_vars = self.optimizer.compute_gradients(self.loss)
                self.grads_and_vars = [[grad, var] for grad, var in self.grads_and_vars if grad is not None]
                # need a global step variable for below?
                self.train_op = self.optimizer.apply_gradients(self.grads_and_vars,
                                                               global_step=tf.contrib.framework.get_global_step())

        # Merge summaries from this network and the shared network (but not the value net)
        var_scope_name = tf.get_variable_scope().name
        summary_ops = tf.get_collection(tf.GraphKeys.SUMMARIES)
        summaries = [s for s in summary_ops if "policy_net" in s.name or "shared" in s.name]
        summaries = summaries + [s for s in summary_ops if var_scope_name in s.name]  # ???
        self.summaries = tf.summary.merge(summaries)


class ValueEstimator:
    """
    Value Function approximator. Returns a value estimator for a batch of observations.

    Args:
        reuse: If true, an existing shared network will be re-used.
        trainable: If true we add train ops to the network.
            Actor threads that don't update their local models and don't need
            train ops would set this to false.
    """

    def __init__(self, reuse=False, trainable=True):
        # Placeholders for our input
        # Our input are 4 RGB frames of shape 160, 160 each  <-- always true?
        self.states = tf.placeholder(shape=[None, 84, 84, 4], dtype=tf.uint8, name="X")
        # The TD target value
        self.targets = tf.placeholder(shape=[None], dtype=tf.float32, name="y")

        X = tf.to_float(self.states) / 255.0

        # Graph shared with Value Net
        with tf.variable_scope("shared", reuse=reuse):
            out = build_shared_network(X, add_summaries=(not reuse))

        with tf.variable_scope("value_net"):
            self.logits = U.dense(out, 1, "logits", U.normc_initializer(1.0))  # ???
            self.logits = tf.squeeze(self.logits, squeeze_dims=[1])

            self.losses = tf.squared_difference(self.logits, self.targets)
            self.loss = tf.reduce_sum(self.losses, name="loss")

            self.predictions = {
                "logits": self.logits
            }

            # Summaries
            # TODO: replace tf summaries with tlogger
            prefix = tf.get_variable_scope().name
            tf.summary.scalar(self.loss.name, self.loss)
            tf.summary.scalar("{}/max_value".format(prefix), tf.reduce_max(self.logits))
            tf.summary.scalar("{}/min_value".format(prefix), tf.reduce_min(self.logits))
            tf.summary.scalar("{}/mean_value".format(prefix), tf.reduce_mean(self.logits))
            tf.summary.scalar("{}/reward_max".format(prefix), tf.reduce_max(self.targets))
            tf.summary.scalar("{}/reward_min".format(prefix), tf.reduce_min(self.targets))
            tf.summary.scalar("{}/reward_mean".format(prefix), tf.reduce_mean(self.targets))
            tf.summary.histogram("{}/reward_targets".format(prefix), self.targets)
            tf.summary.histogram("{}/values".format(prefix), self.logits)

            if trainable:
                # TODO: use optimizer specified in Config
                # self.optimizer = tf.train.AdamOptimizer(1e-4)
                self.optimizer = tf.train.RMSPropOptimizer(0.00025, 0.99, 0.0, 1e-6)
                self.grads_and_vars = self.optimizer.compute_gradients(self.loss)
                self.grads_and_vars = [[grad, var] for grad, var in self.grads_and_vars if grad is not None]
                # need a global step variable for below?
                self.train_op = self.optimizer.apply_gradients(self.grads_and_vars,
                                                               global_step=tf.contrib.framework.get_global_step())

        # why no summaries from the value net?
        var_scope_name = tf.get_variable_scope().name
        summary_ops = tf.get_collection(tf.GraphKeys.SUMMARIES)
        # summaries = [s for s in summary_ops if "policy_net" in s.name or "shared" in s.name]
        summaries = [s for s in summary_ops if "value_net" in s.name or "shared" in s.name]
        summaries = summaries + [s for s in summary_ops if var_scope_name in s.name]  # ???
        self.summaries = tf.summary.merge(summaries)
