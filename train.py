"""Script for training the implementation of alexnet for semantic 
segmentation
"""

import tensorflow as tf
import tensorflow.contrib.slim as slim
import skimage.io as io
import numpy as np
import preprocessor
from PIL import Image
import glob

import fcn

# Graph to preprocess data
p = preprocessor.Processor()
mp = preprocessor.MaskProcessor()

# Placeholder for the inputs
x = tf.placeholder(tf.float32, [None, 227, 227, 3])

# Placeholder for the image mask
y_ = tf.placeholder(tf.float32, [None, 227, 227, 1])

with slim.arg_scope(fcn.alexnet_v2_arg_scope()):
  y_conv = fcn.alexnet_v2(x)

loss = tf.squared_difference(y_, y_conv)
losses = tf.reduce_mean(loss)
train_step = tf.train.AdamOptimizer(1e-4).minimize(losses)

init_op = tf.group(tf.global_variables_initializer(),
                   tf.local_variables_initializer()) 
saver = tf.train.Saver()

images = glob.glob("./train_hq/*.jpg")
masks = ["./train_masks"+x[10:] for x in images]
masks = [x[:-4]+'_mask.gif' for x in masks] 

with tf.Session() as sess:
  sess.run(init_op)

  for i,(im_path,m_path) in enumerate(zip(images, masks)):
    img = np.array(io.imread(im_path))
    mask = np.expand_dims(np.array(io.imread(m_path)), axis=2) 
    img = p.process(sess, img)
    mask = mp.process(sess, mask)
    dmask = Image.fromarray(mask[:,:,0])
    dmask.save("dm{}.png".format(i))
    loss, _ = sess.run([losses, train_step], feed_dict={x:[img], y_:[mask]})
    print("Loss {}".format(loss))
  save_path = saver.save(sess, "/tmp/model.ckpt")  
  print("Model saved in file: {}".format(save_path))

