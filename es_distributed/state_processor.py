# Modified from state_processor.py from Denny Britz's reinforcement-learning project
# Also borrows concepts from OpenAI's universe-starter-agent project
import numpy as np
import tensorflow as tf

class StateProcessor():
    """
    Processes a raw Atari iamges. Resizes it and converts it to grayscale.
    """
    def __init__(self):
        # Build the Tensorflow graph
        with tf.variable_scope("state_processor"):
            self.input_state = tf.placeholder(shape=[210, 160, 3], dtype=tf.uint8)
            self.output = tf.image.rgb_to_grayscale(self.input_state)
            # edited output sizes to match with universe-starter-agent preprocessing
            self.output = tf.image.crop_to_bounding_box(self.output, 34, 0, 160, 160)
            self.output = tf.image.resize_images(
                self.output, [80, 80], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
            self.output = tf.image.resize_images(
                self.output, [42, 42], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
            # self.output = tf.reduce_mean(self.output, reduction_indices=[2])
            # self.output = tf.cast(self.output, tf.float32)
            # self.output *= (1.0 / 255.0)
            self.output = tf.reshape(self.output, [42, 42, 1])
            # self.output = tf.expand_dims(self.output, [0])

    def process(self, state, sess=None):
        """
        Args:
            sess: A Tensorflow session object
            state: A [210, 160, 3] Atari RGB State

        Returns:
            A processed [42, 42, 1] state representing grayscale values.
        """
        sess = sess or tf.get_default_session()
        return sess.run(self.output, {self.input_state: state})