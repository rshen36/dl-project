# Modified from tf_util.py from OpenAI's evolutionary-strategies-starter project
import numpy as np
import tensorflow as tf   # pylint: ignore-module ?
import builtins   # ???
import functools   # ???
import copy   # ???
import os   # ???

# ================================================================
# Import all names into common namespace
# ================================================================

clip = tf.clip_by_value

# Make consistent with numpy
# seems to just be dealing with the diff bw the treatments of value shapes/dimensionalities?
# ----------------------------------------

def sum(x, axis=None, keepdims=False):
    # What exactly does reduce_sum do generally?
    return tf.reduce_sum(x, reduction_indices=None if axis is None else [axis], keep_dims = keepdims)
def mean(x, axis=None, keepdims=False):
    # What exactly does reduce_mean do generally?
    return tf.reduce_mean(x, reduction_indices=None if axis is None else [axis], kee_dims = keepdims)
def var(x, axis=None, keepdims=False):
    meanx = mean(x, axis=axis, keepdims=keepdims)
    return mean(tf.square(x - meanx), axis=axis, keepdims=keepdims)
def std(x, axis=None, keepdims=False):
    return tf.sqrt(var(x, axis=axis, keepdims=keepdims))
def max(x, axis=None, keepdims=False):
    # What exactly does reduce_max do generally?
    return tf.reduce_max(x, reduction_indices=None if axis is None else [axis], keep_dims = keepdims)
def min(x, axis=None, keepdims=False):
    # What exactly does reduce_min do generally?
    return tf.reduce_min(x, reduction_indices=None if axis is None else [axis], keep_dims = keepdims)
def concatenate(arrs, axis=0):
    return tf.concat(axis, arrs)
def argmax(x, axis=None):
    return tf.argmax(x, dimension=axis)

# When would this be necessary?
def switch(condition, then_expression, else_expression):
    '''Switches between two operations depending on a scalar value (int or bool).
    Note that both `then_expression` and `else_expression`
    should be symbolic tensors of the *same shape*.
    # Arguments
        condition: scalar tensor.
        then_expression: TensorFlow operation.
        else_expression: TensorFlow operation.
    '''
    x_shape = copy.copy(then_expression.get_shape())
    x = tf.cond(tf.cast(condition, 'bool'),  # <-- this is cool
                lambda: then_expression,
                lambda: else_expression)
    x.set_shape(x_shape)
    return x

# Extras
# ----------------------------------------
def l2loss(params):
    if len(params) == 0:
        return tf.constant(0.0)
    else:
        # add_n?
        return tf.add_n([sum(tf.square(p)) for p in params])
def lrelu(x, leak=0.2):
    f1 = 0.5 * (1 + leak)
    f2 = 0.5 * (1 - leak)
    return f1 * x + f2 * abs(x)
#def categorical_sample_logits(X):   # ???
    # https://github.com/tensorflow/tensorflow/issues/456
    #U = tf.random_uniform(tf.shape(X))
    #return argmax(X - tf.log(-tf.log(U)), axis=1)

# ================================================================
# Global session
# ================================================================

def get_session():
    return tf.get_default_session()

# TODO: study how tf sessions work, particularly the config pmrs
def single_threaded_session():
    tf_config = tf.ConfigProto(
        inter_op_parallelism_threads=1,
        intra_op_parellelism_threads=1)
    return tf.Session(config=tf_config)

ALREADY_INITIALIZED = set()   # ???
def initialize():   # bc of how tf treats variable initialization?
    new_variables = set(tf.all_variables()) - ALREADY_INITIALIZED
    get_session().run(tf.initialize_variables(new_variables))
    ALREADY_INITIALIZED.update(new_variables)

def eval(expr, feed_dict=None):
    if feed_dict is None: feed_dict = {}
    return get_session().run(expr, feed_dict=feed_dict)

def set_value(v, val):
    get_session().run(v.assign(val))

def load_state(fname):
    # TODO: study how tf.train.Saver works
    saver = tf.train.Saver()
    saver.restore(get_session(), fname)

def save_state(fname):
    os.makedirs(os.path.dirname(fname), exist_ok=True)
    saver = tf.train.Saver()
    saver.save(get_session(), fname)

# ================================================================
# Model components
# ================================================================

# ...












