"""Script for training the implementation of alexnet for semantic 
segmentation
"""

import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np

import fcn

tfrecords_filename = 'debug_data.tfrecords'

def read_and_decode(filename_queue):
  """Same reference as preprocessor.py."""
  reader = tf.TFRecordReader()

  _, serialized_example = reader.read(filename_queue)

  features = tf.parse_single_example(
          serialized_example,
          features={
              'height': tf.FixedLenFeature([], tf.int64),
              'width': tf.FixedLenFeature([], tf.int64),
              'image_raw': tf.FixedLenFeature([], tf.string),
              'mask_raw': tf.FixedLenFeature([], tf.string)
          })
  image = tf.decode_raw(features['image_raw'], tf.uint8)
  mask = tf.decode_raw(features['mask_raw'], tf.uint8)

  height = tf.cast(features['height'], tf.int32)
  width = tf.cast(features['width'], tf.int32)

  image_shape = tf.stack([height, width, 3])
  mask_shape = tf.stack([height, width, 1])

  image = tf.reshape(image, image_shape)
  mask = tf.reshape(mask, mask_shape)

  image_size_const = tf.constant((227, 227, 3), dtype=tf.int32)
  mask_size_const = tf.constant((227, 227, 1), dtype=tf.int32)

  resized_image = tf.image.resize_images(image, [227, 227])
  resized_mask = tf.image.resize_images(image, [227, 227])
  images, masks = tf.train.shuffle_batch( [resized_image, resized_mask],
          batch_size=2, capacity=30, num_threads=2, min_after_dequeue=10)
  return images, masks

# Set up data pipeline
filename_queue = tf.train.string_input_producer(
    [tfrecords_filename], num_epochs=10)

image, mask = read_and_decode(filename_queue)
# Placeholder for the inputs
x = tf.placeholder(tf.float32, [None, 227, 227, 3])

# Placeholder for the image mask
y_ = tf.placeholder(tf.float32, [None, 227, 227, 3])

with slim.arg_scope(fcn.alexnet_v2_arg_scope()):
  y_conv = fcn.alexnet_v2(x)

loss = tf.squared_difference(y_, y_conv)
losses = tf.reduce_mean(loss)
train_step = tf.train.AdamOptimizer(1e-4).minimize(losses)

init_op = tf.group(tf.global_variables_initializer(),
                   tf.local_variables_initializer()) 
saver = tf.train.Saver()

with tf.Session() as sess:
  sess.run(init_op)

  coord = tf.train.Coordinator()
  threads = tf.train.start_queue_runners(coord=coord)
  for i in range(340):
    img, m = sess.run([image, mask]) 
    loss, _ = sess.run([losses, train_step], feed_dict={x:img, y_:m})
    print("Loss {}".format(loss))
  save_path = saver.save(sess, "/tmp/model.ckpt")  
  print("Model saved in file: {}".format(save_path))

