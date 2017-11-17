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
  """AlexNet version 2.

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

  #TODO replace the max_pool2d layers with strided convolutional layers
  net = layers.conv2d(
      inputs, 96, [11, 11], 4, padding='VALID', scope='conv1')
  net = layers_lib.max_pool2d(net, [3, 3], 2, scope='pool1') 
  net = layers.conv2d(net, 192, [5, 5], scope='conv2')
  net = layers_lib.max_pool2d(net, [3, 3], 2, scope='pool2')
  net = layers.conv2d(net, 384, [3, 3], scope='conv3')
  net = layers.conv2d(net, 384, [3, 3], scope='conv4')
  net = layers.conv2d(net, 256, [3, 3], scope='conv5')

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
    # Replace classification layer with convolution to prediction at 
    # course output points (create pixelwise classification on the
    # currently downsampled image).
    net = layers.conv2d(
        net,
        2, [1,1], # Prediction is either 'car' or 'background' for Carvana.
        padding='VALID',
        biases_initializer=init_ops.zeros_initializer(),
        scope='fc8') 

  # Deconvolution net
  with arg_scope(
      [layers.conv2d_transpose],
      padding='VALID'):
    net = layers.conv2d_transpose(net, 4096, 1, scope='convt8')
    net = layers.conv2d_transpose(net, 4096, 1, scope='convt7')
    net = layers.conv2d_transpose(net, 256, 5, scope='convt6')
    net = layers.conv2d_transpose(net, 384, 3, scope='convt5')
    net = layers.conv2d_transpose(net, 384, 3, scope='convt4')
    net = layers.conv2d_transpose(net, 192, 3, scope='convt3') # Technically should be an unpool w/ switch until the pool->strided conv is implemented
    net = layers.conv2d_transpose(net, 192, 5, scope='convt2')
    net = layers.conv2d_transpose(net, 96, 5, scope='convt1') # Technically should be an unpool as well
    net = layers.conv2d_transpose(net, 3, 11, 4, scope='convt0') 

  return net

