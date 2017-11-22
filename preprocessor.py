"""Script to convert the data into a tfrecords file. Modified from tutorial
by warmspringwinds.github.io/tensorflow/tf-slim/2016/12/21/tfrecords-guide.
"""
import glob
import tensorflow as tf
import skimage.io as io
import numpy as np

original_images = []

def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

debug_im = glob.glob("./train_hq/*.jpg")
debug_im = debug_im[:30] # Use small subset for quicker training
debug_mask = ['./train_masks'+x[10:] for x in debug_im]
debug_mask = [x[:-4]+'_mask.gif' for x in debug_mask]
tfrecords_filename = 'debug_data.tfrecords'

writer = tf.python_io.TFRecordWriter(tfrecords_filename)

for im,m in zip(debug_im, debug_mask):
    img = np.array(io.imread(im))
    mask = np.array(io.imread(m))

    height = img.shape[0]
    width = img.shape[1]

    img_raw = img.tostring()
    mask_raw = mask.tostring()

    example = tf.train.Example(features=tf.train.Features(feature={
        'height': _int64_feature(height),
        'width': _int64_feature(width),
        'image_raw': _bytes_feature(img_raw),
        'mask_raw': _bytes_feature(mask_raw)}))

    writer.write(example.SerializeToString())
writer.close()

