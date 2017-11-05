"""Modified AlexNet model definition from 
https://github.com/tensorflow/tensorflow/blob/fa4ba830f437fdb9dc1085b4d68a3bab41a16e20/tensorflow/contrib/slim/python/slim/nets/alexnet.py

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
      [layers.conv2d, layers_lib.fully_connected],
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
        To use in classification mode, resize input to 224x224. To use in fully
        convolutional mode, set spatial_squeeze to false.
        The LRN layers have been removed and change the initializers from
        random_normal_initializer to xavier_initializer.

  Args:
    inputs: a tensor of size [batch_size, height, width, channels].
    is_training: whether or not the model is being trained.
    dropout_keep_prob: the probability that activations are kept in the dropout
      layers during training.
    scope: Optional scope for the variables.

  Returns:
    the last op containing the log predictions and end_points dict.
  """
  with variable_scope.variable_scope(scope, 'alexnet_v2', [inputs]) as sc:
    end_points_collection = sc.original_name_scope + '_end_points'

    #TODO replace the max_pool2d layers with strided convolutional layers
    # Collect outputs for conv2d, fully_connected and max_pool2d.
    with arg_scope(
        [layers.conv2d, layers_lib.fully_connected, layers_lib.max_pool2d],
        outputs_collections=[end_points_collection]): 
        #TODO use end_points_collection to fuse conv outputs as fcn8,16,32?
      net = layers.conv2d(
          inputs, 96, [11, 11], 4, padding='VALID', scope='conv1')
      net = layers_lib.max_pool2d(net, [3, 3], 2, scope='pool1') 
      net = layers.conv2d(net, 192, [5, 5], scope='conv2')
      net = layers_lib.max_pool2d(net, [3, 3], 2, scope='pool2')
      net = layers.conv2d(net, 384, [3, 3], scope='conv3')
      net = layers.conv2d(net, 384, [3, 3], scope='conv4')
      net = layers.conv2d(net, 256, [3, 3], scope='conv5')

      # Use conv2d instead of fully_connected layers.
      with arg_scope(
          [layers.conv2d],
          weights_initializer=trunc_normal(0.005),
          biases_initializer=init_ops.constant_initializer(0.1)):

        #TODO consider less downsampling given the large input image
        net = layers.conv2d(net, 4096, [5, 5], padding='VALID', scope='fc5')
        net = layers_lib.dropout(
            net, dropout_keep_prob, is_training=is_training, scope='dropout5')
        net = layers.conv2d(net, 4096, [1, 1], scope='fc6')
        net = layers_lib.dropout(
            net, dropout_keep_prob, is_training=is_training, scope='dropout6')
        # Replace classification layer with convolution to prediction at 
        # course output points (create pixelwise classification on the
        # currently downsampled image).
        net = layers.conv2d(
            net,
            2, [1,1], # Prediction is either 'car' or 'background' for Carvana.
            padding='VALID',
            biases_initializer=init_ops.zeros_initializer(),
            scope='fc8') 

        # TODO Upsample the course output points to create a image mask of the
        # same size as the input image. Compare using multiple up steps like in 
        # u-net to one large up step.

      # Convert end_points_collection into a end_point dict.
      return net, end_points


alexnet_v2.default_image_size = 224
