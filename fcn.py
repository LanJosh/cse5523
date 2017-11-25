"""Modified AlexNet model definition from 
https://github.com/tensorflow/models/blob/master/research/slim/nets/alexnet.py

This work was first described in:
  ImageNet Classification with Deep Convolutional Neural Networks
  Alex Krizhevsky, Ilya Sutskever and Geoffrey E. Hinton

and later refined in:
  One weird trick for parallelizing convolutional neural networks
  Alex Krizhevsky, 2014

Here we provide the implementation proposed in "One weird trick" modified to 
use strided convolutional layers instead of maxpool layers as described in:
  Striving for Simplicity: The All Convolutional Net
  Jost Tobias Springenber, Alexey Dosovitskiy, Thomas Brox, Martin Riedmiller, 2015

as well as modify the network to perform semantic segmentation instead of image 
classification as described in:
  Fully Convolutional Networks for Semantic Segmentation
  Jonathon Long, Evan Shelhamer, Trevor Darrell

The '2nd' part of the network is a small deconvolutional network inspired by, but 
does not require usage of unpooling as each pooling layer is replaced with a 
strided convolutional layer:
  Learning Deconvolution Network for Semantic Segmentation
  Hyeonwoo Noh, Seunghoon Hong, Bohyung Han

Usage:
  with slim.arg_scope(alexnet.alexnet_v2_arg_scope()):
    outputs, end_points = alexnet.alexnet_v2(inputs)

@@alexnet_v2
"""

import tensorflow as tf
from tensorflow.contrib import layers
from tensorflow.contrib.framework.python.ops import arg_scope
from tensorflow.contrib.layers.python.layers import layers as layers_lib
from tensorflow.contrib.layers.python.layers import regularizers
from tensorflow.contrib.layers.python.layers import utils
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import variable_scope

trunc_normal = lambda stddev: init_ops.truncated_normal_initializer(0.0, stddev)


def alexnet_v2_arg_scope(weight_decay=0.0005):
  with arg_scope(
      [layers.conv2d],
      activation_fn=nn_ops.relu,
      biases_initializer=init_ops.constant_initializer(0.1),
      weights_regularizer=regularizers.l2_regularizer(weight_decay)):
    with arg_scope([layers.conv2d], padding='SAME'):
      with arg_scope([layers_lib.max_pool2d], padding='VALID') as arg_sc:
        return arg_sc


def alexnet_v2(inputs,
               is_training=True,
               dropout_keep_prob=0.5,
               scope='alexnet_v2'):
  """Modified version of AlexNet version 2 with a deconvolutional expanding
  path for semantic segmentation.

  Described in: http://arxiv.org/pdf/1404.5997v2.pdf

  Note: All the fully_connected layers have been transformed to conv2d layers.

  Args:
    inputs: a tensor of size [batch_size, 227, 227, 3].
    is_training: whether or not the model is being trained.
    dropout_keep_prob: the probability that activations are kept in the dropout
      layers during training.
    scope: Optional scope for the variables.

  Returns:
    The last layer containing a segmentation map of an image.
  """

  net = layers.conv2d(
      inputs, 96, [11, 11], 4, padding='VALID', scope='conv1')
  net = layers.conv2d(net, 192, 3, 2, padding='VALID', scope='pconv1')
  net = layers.conv2d(net, 192, [5, 5], padding='VALID', scope='conv2')
  net = layers.conv2d(net, 384, 3, 2, padding='VALID', scope='pconv2')
  net = layers.conv2d(net, 384, [3, 3], padding='VALID', scope='conv3')
  net = layers.conv2d(net, 384, [3, 3], padding='VALID', scope='conv4')
  net = layers.conv2d(net, 256, [3, 3], padding='VALID', scope='conv5')

  # Convolution net
  with arg_scope(
      [layers.conv2d],
      weights_initializer=trunc_normal(0.005),
      biases_initializer=init_ops.constant_initializer(0.1)):

    net = layers.conv2d(net, 4096, [5, 5], padding='VALID', scope='fc6')
    net = layers_lib.dropout(
        net, dropout_keep_prob, is_training=is_training, scope='dropout6')
    net = layers.conv2d(net, 4096, [1, 1], scope='fc7')
    net = layers_lib.dropout(
        net, dropout_keep_prob, is_training=is_training, scope='dropout7')
    net = layers.conv2d(
        net,
        2, [1,1], # Prediction is either 'car' or 'background' for Carvana.
        padding='VALID',
        activation_fn=tf.nn.sigmoid,
        biases_initializer=init_ops.zeros_initializer(),
        scope='fc8') 

  # Deconvolution net
  with arg_scope(
      [layers.conv2d_transpose],
      padding='VALID',
      activation_fn=nn_ops.relu):
    net = layers.conv2d_transpose(net, 4096, 1, scope='convt9')
    net = layers.conv2d_transpose(net, 4096, 1, scope='convt10')
    net = layers.conv2d_transpose(net, 256, 5, scope='convt11')
    net = layers.conv2d_transpose(net, 384, 3, scope='convt12')
    net = layers.conv2d_transpose(net, 384, 3, scope='convt13')
    net = layers.conv2d_transpose(net, 384, 3, scope='convt14')
    net = layers.conv2d_transpose(net, 192, 3, 2, scope='convt15')
    net = layers.conv2d_transpose(net, 192, 5, scope='convt16')
    net = layers.conv2d_transpose(net, 96, 3, 2, scope='convt17')
    net = layers.conv2d_transpose(net, 2, 11, 4, activation_fn=tf.nn.sigmoid, scope='convt18')

  return net

def alexnet_v2_orig(inputs,
               is_training=True,
               dropout_keep_prob=0.5,
               scope='alexnet_v2'):
  """Original AlexNet version 2 as described by Long,Shelhamer for semantic
  segmentation.

  Described in: http://arxiv.org/pdf/1404.5997v2.pdf

  Note: All the fully_connected layers have been transformed to conv2d layers.

  Args:
    inputs: a tensor of size [batch_size, 227, 227, 3].
    is_training: whether or not the model is being trained.
    dropout_keep_prob: the probability that activations are kept in the dropout
      layers during training.
    scope: Optional scope for the variables.

  Returns:
    The last layer containing a segmentation map of an image.
  """

  net = layers.conv2d(
      inputs, 96, [11, 11], 4, padding='VALID', scope='conv1')
  net = layers.max_pool2d(net, [3,3], 2, scope='pool1')
  net = layers.conv2d(net, 192, [5, 5], padding='VALID', scope='conv2')
  net = layers.max_pool2d(net, [3,3], 2, scope='pool2')
  net = layers.conv2d(net, 384, [3, 3], padding='VALID', scope='conv3')
  net = layers.conv2d(net, 384, [3, 3], padding='VALID', scope='conv4')
  net = layers.conv2d(net, 256, [3, 3], padding='VALID', scope='conv5')
  net = layers.max_pool2d(net, [3,3], 2, scope='pool3')

  # Convolution net
  with arg_scope(
      [layers.conv2d],
      weights_initializer=trunc_normal(0.005),
      biases_initializer=init_ops.constant_initializer(0.1)):

    net = layers.conv2d(net, 4096, [2, 2], padding='VALID', scope='fc6')
    net = layers_lib.dropout(
        net, dropout_keep_prob, is_training=is_training, scope='dropout6')
    net = layers.conv2d(net, 4096, [1, 1], scope='fc7')
    net = layers_lib.dropout(
        net, dropout_keep_prob, is_training=is_training, scope='dropout7')
    net = layers.conv2d(
        net,
        2, [1,1], # Prediction is either 'car' or 'background' for Carvana.
        activation_fn=tf.nn.sigmoid,
        padding='VALID',
        biases_initializer=init_ops.zeros_initializer(),
        scope='fc8') 

  # Deconvolution net
  with arg_scope(
      [layers.conv2d_transpose],
      padding='VALID'):
    net = layers.conv2d_transpose(net, 2, 227, 32, activation_fn=tf.nn.sigmoid, scope='deconv')

  return net

