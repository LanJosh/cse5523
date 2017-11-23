import glob
import tensorflow as tf
import skimage.io as io
import numpy as np

class Processor():
    """Resizes the images to be square 227x227 RGB images."""

    def __init__(self):
        self.input = tf.placeholder(shape=[1280, 1918, 3], dtype=tf.uint8)
        self.output = tf.image.resize_images(
            self.input, [227, 227],
            method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

    def process(self, sess, image):
        return sess.run(self.output, {self.input:image})

class MaskProcessor():
    def __init__(self):
        self.input = tf.placeholder(shape=[1280, 1918, 1], dtype=tf.uint8)
        # Hack w/ only 2 classes means we skip probability scores and
        # try the loss instead of with logits to be the actual pixel value
        #self.output = tf.divide(self.input, tf.constant(255, tf.uint8))
        self.output = tf.image.resize_images(
            self.input, [227, 227],
            method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

    def process(self, sess, image):
        return sess.run(self.output, {self.input:image})
