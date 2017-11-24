import glob
import tensorflow as tf
import skimage.io as io
import numpy as np

class Processor():
    """Resizes the images to be square 227x227 RGB images."""

    def __init__(self):
        self.input = tf.placeholder(shape=[1280, 1918, 3], dtype=tf.int8)
        self.output = tf.image.resize_images(
            self.input, [227, 227],
            method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

    def process(self, sess, image):
        return sess.run(self.output, {self.input:image})

class MaskProcessor():
    def __init__(self):
        self.input = tf.placeholder(
                shape=[None, 1280, 1918, 1],dtype=tf.int8)
        self.output = tf.image.resize_images(
                self.input, [227, 227],
                method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

        self.car_sc = tf.divide(self.output,tf.constant(255,dtype=tf.int8))
        self.bg_sc = tf.squeeze(self.car_sc)
        self.car_sc = tf.squeeze(self.car_sc)

        self.bg_sc = tf.subtract(self.bg_sc, tf.constant(1.))
        self.bg_sc = tf.multiply(self.bg_sc, tf.constant(-1.))
        self.output = tf.stack([self.car_sc, self.bg_sc], axis=2)

    def process(self, sess, image):
        return sess.run(self.output, {self.input:image})
