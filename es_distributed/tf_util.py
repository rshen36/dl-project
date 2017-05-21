# Modified from tf_util.py from OpenAI's evolutionary-strategies-starter project
import numpy as np
import tensorflow as tf   # pylint: ignore-module ?
import copy
import os

# ================================================================
# Import all names into common namespace
# ================================================================

clip = tf.clip_by_value

# Make consistent with numpy
# seems to just be dealing with the diff bw the treatments of value shapes/dimensionalities?
# ----------------------------------------

def sum(x, axis=None, keepdims=False):
    return tf.reduce_sum(x, reduction_indices=None if axis is None else [axis], keep_dims=keepdims)
def mean(x, axis=None, keepdims=False):
    return tf.reduce_mean(x, reduction_indices=None if axis is None else [axis], keep_dims=keepdims)
def var(x, axis=None, keepdims=False):
    meanx = mean(x, axis=axis, keepdims=keepdims)
    return mean(tf.square(x - meanx), axis=axis, keepdims=keepdims)
def std(x, axis=None, keepdims=False):
    return tf.sqrt(var(x, axis=axis, keepdims=keepdims))
def max(x, axis=None, keepdims=False):
    return tf.reduce_max(x, reduction_indices=None if axis is None else [axis], keep_dims=keepdims)
def min(x, axis=None, keepdims=False):
    return tf.reduce_min(x, reduction_indices=None if axis is None else [axis], keep_dims=keepdims)
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
# def categorical_sample_logits(X):  # what's the diff bw this and below?
#     # https://github.com/tensorflow/tensorflow/issues/456
#     U = tf.random_uniform(tf.shape(X))
#     return argmax(X - tf.log(-tf.log(U)), axis=1)
def categorical_sample(logits, d):
    value = tf.squeeze(tf.multinomial(logits - max(logits, [1], keepdims=True), [1]))
    return tf.one_hot(value, d)

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
    new_variables = set(tf.global_variables()) - ALREADY_INITIALIZED
    get_session().run(tf.variables_initializer(new_variables))
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

# normalized columns initializer
def normc_initializer(std=1.0):
    def _initializer(shape, dtype=None, partition_info=None):  # pylint: disable=W0613 ?
        out = np.random.randn(*shape).astype(np.float32)
        out *= std / np.sqrt(np.square(out).sum(axis=0, keepdims=True))
        return tf.constant(out)
    return _initializer


def dense(x, size, name, weight_init=None, bias=True):
    if len(x.get_shape()) > 2:
        shape = x.get_shape().as_list()
        x = tf.reshape(x, [shape[0], shape[1] * shape[2]])
    w = tf.get_variable(name + "/w", [x.get_shape()[1], size], initializer=weight_init)
    ret = tf.matmul(x, w)

    if bias:
        b = tf.get_variable(name + "/b", [size], initializer=tf.constant_initializer(0.0))
        return ret + b
    else:
        return ret


def conv2d(x, num_filters, name, filter_size=(3, 3), stride=(1, 1), pad="SAME", dtype=tf.float32, collections=None):
    with tf.variable_scope(name):
        stride_shape = [1, stride[0], stride[1], 1]
        filter_shape = [filter_size[0], filter_size[1], int(x.get_shape()[3]), num_filters]

        # there are "num input feature maps * filter height * filter width"
        # inputs to each hidden unit
        fan_in = np.prod(filter_shape[:3])
        # each unit in the lower layer receives a gradient from:
        # "num output feature maps * filter height * filter width" /
        #   pooling size
        fan_out = np.prod(filter_shape[:2]) * num_filters
        # initialize weights with random weights
        w_bound = np.sqrt(6. / (fan_in + fan_out))

        w = tf.get_variable("W", filter_shape, dtype, tf.random_uniform_initializer(-w_bound, w_bound),
                            collections=collections)
        b = tf.get_variable("b", [1, 1, 1, num_filters], initializer=tf.constant_initializer(0.0),
                            collections=collections)
        return tf.nn.conv2d(x, w, stride_shape, pad) + b

# ================================================================
# Basic Stuff
# ================================================================

def function(inputs, outputs, updates=None, givens=None):
    if isinstance(outputs, list):
        return _Function(inputs, outputs, updates, givens=givens)
    elif isinstance(outputs, dict):
        f = _Function(inputs, outputs.values(), updates, givens=givens)
        return lambda *inputs : dict(zip(outputs.keys(), f(*inputs)))
    else:
        f = _Function(inputs, [outputs], updates, givens=givens)
        return lambda *inputs : f(*inputs)[0]


# how tf does this work...
class _Function(object):
    def __init__(self, inputs, outputs, updates, givens, check_nan=False):
        assert all(len(i.op.inputs) == 0 for i in inputs), "inputs should all be placeholders"
        self.inputs = inputs
        updates = updates or []
        self.update_group = tf.group(*updates)   # group?
        self.outputs_update = list(outputs) + [self.update_group]
        self.givens = {} if givens is None else givens
        self.check_nan = check_nan
    def __call__(self, *inputvals):  # ???
        assert len(inputvals) == len(self.inputs)
        feed_dict = dict(zip(self.inputs, inputvals))
        feed_dict.update(self.givens)
        results = get_session().run(self.outputs_update, feed_dict=feed_dict)[:-1]
        if self.check_nan:
            if any(np.isnan(r).any() for r in results):
                raise RuntimeError("Nan detected")
        return results

# ================================================================
# Graph traversal
# ================================================================

VARIABLES = {}

# ================================================================
# Flat vectors
# ================================================================

def var_shape(x):
    out = x.get_shape().as_list()
    assert all(isinstance(a, int) for a in out), "shape function assumes that shape is fully known"
    return out

def numel(x):
    return intprod(var_shape(x))

def intprod(x):
    return int(np.prod(x))

def flatgrad(loss, var_list):
    grads = tf.gradients(loss, var_list)
    return tf.concat(0, [tf.reshape(grad, [numel(v)]) for (v, grad) in zip(var_list, grads)])


class SetFromFlat(object):
    def __init__(self, var_list, dtype=tf.float32):
        shapes = list(map(var_shape, var_list))
        total_size = np.sum([intprod(shape) for shape in shapes])

        self.theta = theta = tf.placeholder(dtype, [total_size])
        start = 0
        assigns = []
        for (shape, v) in zip(shapes, var_list):
            size = intprod(shape)
            assigns.append(tf.assign(v, tf.reshape(theta[start:start+size],shape)))   # ???
            start += size
        assert start == total_size
        self.op = tf.group(*assigns)
    def __call__(self, theta):
        get_session().run(self.op, feed_dict={self.theta:theta})


class GetFlat(object):
    def __init__(self, var_list):
        self.op = tf.concat([tf.reshape(v, [numel(v)]) for v in var_list], 0)  # for tf 0.12
    def __call__(self):
        return get_session().run(self.op)

# ================================================================
# Misc
# ================================================================

def scope_vars(scope, trainable_only):
    """
    Get variables inside a scope
    The scope can be specified as a string
    """
    return tf.get_collection(
        tf.GraphKeys.TRAINABLE_VARIABLES if trainable_only else tf.GraphKeys.VARIABLES,
        scope=scope if isinstance(scope, str) else scope.name
    )

def in_session(f):
    @function.wraps(f)   # ???
    def newfunc(*args, **kwargs):
        with tf.Session():
            f(*args, **kwargs)
    return newfunc


# why is the cache necessary?
_PLACEHOLDER_CACHE = {}    # name -> (placeholder, dtype, shape)
def get_placeholder(name, dtype, shape):
    print("calling get_placeholder", name)
    if name in _PLACEHOLDER_CACHE:
        out, dtype1, shape1 = _PLACEHOLDER_CACHE[name]
        assert dtype1==dtype and shape1==shape
        return out
    else:
        out = tf.placeholder(dtype=dtype, shape=shape, name=name)
        _PLACEHOLDER_CACHE[name] = (out,dtype,shape)
        return out
def get_placeholder_cached(name):
    return _PLACEHOLDER_CACHE[name][0]


def flatten(x):
    return tf.reshape(x, [-1, intprod(x.get_shape().as_list()[1:])])

def reset():
    global _PLACEHOLDER_CACHE
    global VARIABLES
    _PLACEHOLDER_CACHE = {}
    VARIABLES = {}
    tf.reset_default_graph()