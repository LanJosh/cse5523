"""Script for training the implementation of alexnet for semantic 
segmentation
"""

import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np

import fcn


# Placeholder for the inputs
x = tf.placeholder(tf.float32, [None, 227, 227, 3])

# Placeholder for the image mask
y_ = tf.placeholder(tf.float32, [None, 227, 227, 3])

xd = np.ones((1, 227, 227, 3))
yd = np.zeros((1, 227, 227, 3))

with slim.arg_scope(fcn.alexnet_v2_arg_scope()):
  y_conv = fcn.alexnet_v2(x)

loss = tf.squared_difference(y_, y_conv)
losses = tf.reduce_mean(loss)
train_step = tf.train.AdamOptimizer(1e-4).minimize(losses)

with tf.Session() as sess:
  sess.run(tf.global_variables_initializer())
  train_step.run(feed_dict={x: xd, y_: yd})

