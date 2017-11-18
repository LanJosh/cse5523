"""Script for training the implementation of alexnet for semantic 
segmentation
"""

import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np

import fcn

# TODO @Thaight27 this should result in an array of tuples where the first 
# element is the image to mask and the second element is the ground truth mask
data = preprocessing.get_data()

x = tf.placeholder(tf.float32, [None, 227, 227, 3])
y_ = tf.placeholder(tf.float32, [None, 227, 227, 3])
with slim.arg_scope(fcn.alexnet_v2_arg_scope()):
  y_conv = alexnet_v2(x)
loss = tf.squared_difference(y_, y_conv)
losses = tf.reduce_mean(loss)
train_step = tf.train.AdamOptimizer(1e-4).minimize(losses)

# TODO Implement some saving of model parameters for writeup, print the loss 
# for debugging
with tf.Session() as sess:
  sess.run(tf.global_variables_initializer())
  for im,mask in data:
      train_step.run(feed_dict={x: im, y: mask})

