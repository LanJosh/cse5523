"""Script for calculating the test accuracy of the trained model."""

import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np
import skimage.io as io
import glob
import fcn
import preprocessor
from PIL import Image


test_imgs = glob.glob("./train_hq/*.jpg")
images = test_imgs[-1018:]
ground_truths = ["./train_masks"+x[10:] for x in images]
ground_truths = [x[:-4]+'_mask.gif' for x in ground_truths]

p = preprocessor.Processor()
mp = preprocessor.MaskProcessor()

x = tf.placeholder(tf.float32, (None, 227, 227, 3))
with slim.arg_scope(fcn.alexnet_v2_arg_scope()):
    y_c = fcn.alexnet_v2_orig(x, is_training=False)

# Placeholder for prediction mask
pred = tf.placeholder(tf.float32, (None, 227, 227, 2))
# Placeholder for actual mask
actual = tf.placeholder(tf.float32, (None, 227, 227, 2))
dice_coeff = tf.multiply(pred, actual)
dice_coeff = tf.reduce_sum(dice_coeff)
dice_coeff = tf.multiply(dice_coeff, 2)
dice_coeff = tf.divide(dice_coeff, 206166)

with tf.Session() as sess:
    saver = tf.train.Saver()
    saver.restore(sess, tf.train.latest_checkpoint('/tmp/orig/'))
    accuracy = 0
    for idx,(test_img,ground_truth) in enumerate(zip(test_imgs, ground_truths)):
        # Get pixelwise prediction maps 
        img = np.array(io.imread(test_img))
        img = p.process(sess, img) 
        q = sess.run(y_c, feed_dict={x:[img]})
        q = q[0,:,:,:]

        # Convert pixelwise prediction maps to masked image
        car_scores = q[:,:,0]
        bg_scores = q[:,:,1]
        mask = np.empty((227, 227,2))
        for i in range(227):
            for j in range(227):
                if car_scores[i][j] > bg_scores[i][j]:
                    mask[i][j][0] = 1
                    mask[i][j][1] = 0
                else:
                    mask[i][j][0] = 0
                    mask[i][j][1] = 1
        gt = np.expand_dims(np.array(io.imread(ground_truth)), axis=2)
        gt = mp.process(sess, np.array([gt]))

        # Compute the dice coefficient between the predicted mask and
        # the actual mask
        accuracy += sess.run(dice_coeff, {pred:[mask], actual:[gt]}) 
        print("Accuracy: {}".format(accuracy / (idx+1)))
print("Accuracy on Test Set {}".format(accuracy / 1018.)) 
#print(accuracy)
