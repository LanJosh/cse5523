import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np
import skimage.io as io
import fcn
from PIL import Image

test_img = io.imread('./train_hq/0ed6904e1004_13.jpg') 
with tf.Session() as sess:
    img = tf.image.resize_images(test_img, [227, 227])
    img = img.eval(session=sess) 

x = tf.placeholder(tf.float32, (None, 227, 227, 3))
with slim.arg_scope(fcn.alexnet_v2_arg_scope()):
    y_c = fcn.alexnet_v2_orig(x, is_training=False)

with tf.Session() as sess:
    saver = tf.train.Saver()
    saver.restore(sess, tf.train.latest_checkpoint('/tmp/orig/'))
    q = sess.run(y_c, feed_dict={x:[img]}) 
q = q[0,:,:,:]
car_scores = q[:,:,0]
bg_scores = q[:,:,1]
mask = np.empty((227, 227))
for i in range(227):
    for j in range(227):
        if car_scores[i][j] > bg_scores[i][j]:
            mask[i][j] = 255
        else:
            mask[i][j] = 0
mask = Image.fromarray(mask.astype('uint8'))
mask.save('ita1.png')
