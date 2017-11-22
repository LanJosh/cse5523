# processing Carvana image dataset
# All input data should be kept in a folder called 'input'

import skimage
from skimage import data
from scipy import ndimage
import numpy as np
import pandas as pd
from pandas import DataFrame
import matplotlib.pylab as plt
import os
from glob import glob

INPUT_PATH = '../input' #Get the paths to the input data
DATA_PATH = INPUT_PATH

TRAIN_DATA = os.path.join(DATA_PATH, "train")
TRAIN_MASKS_DATA = os.path.join(DATA_PATH, "train_masks")
#TEST_DATA = os.path.join(DATA_PATH, "test") #comment out until actual testing
TRAIN_MASKS_CSV_PATH = os.path.join(DATA_PATH, "train_masks.csv")
METADATA_CSV_PATH = os.path.join(DATA_PATH, "metadata.csv")

TRAIN_MASKS_CSV = pd.read_csv(TRAIN_MASKS_CSV_PATH) #Read DataFrames
METADATA_CSV = pd.read_csv(METADATA_CSV_PATH)

train_files = glob(os.path.join(TRAIN_DATA, "*.jpg")) #Get paths for each image
train_ids = [s[len(TRAIN_DATA)+1:-4] for s in train_files] #Get the id's by removing ../input/train/ and .jpg

"""
# Get the test image files and their id's into lists
test_files = glob(os.path.join(TEST_DATA, "*.jpg"))
test_ids = [s[len(TEST_DATA)+1:-4] for s in test_files]
"""

def train_mask_index(image_id):
    for i in range(0,TRAIN_MASKS_CSV['img'].size):
        if TRAIN_MASKS_CSV['img'].get_value(i) == str(image_id+'.jpg'):
            return i

def rle_mask(image_id):
    index = train_mask_index(image_id)
    return TRAIN_MASKS_CSV['rle_mask'].get_value(index)

#Returns the path of the filename for the image id
def get_filename(image_id, image_type):
    suffix = ''
    ext = 'jpg'
    if "Train" == image_type:
        data_path = TRAIN_DATA
    elif "Train_mask" == image_type:
        ext = 'gif'
        data_path = TRAIN_MASKS_DATA
        suffix = '_mask'
    elif "Test" == image_type:
        data_path = TEST_DATA
    else:
        raise Exception("Image type '%s' is not recognized" % image_type)

    if not os.path.exists(data_path):
        raise Exception("Data path '%s' does not exists" % data_path)
    
    return os.path.join(data_path, "{}{}.{}".format(image_id, suffix, ext))


#Returns image data 
def get_image_data(image_id, image_type, **args):
    fname = get_filename(image_id, image_type)
    if 'mask' in image_type:
        img = ndimage.imread(fname, mode = 'L')
        assert img is not None, "Failed to read image : %s, %s" % (image_id, image_type)
        img[img <= 127] = 0
        img[img > 127] = 1
    else:
        img = ndimage.imread(fname)
        assert img is not None, "Failed to read image : %s, %s" % (image_id, image_type)
    return img


# Display first training image
image_id = train_ids[500]
img = get_image_data(image_id, "Train")
plt.imshow(img)
plt.show()

# Display first training mask image
mask = get_image_data(image_id, "Train_mask")
plt.imshow(mask, cmap = 'Greys_r')
plt.show()

print("Image shape: {} | image type: {} | mask shape: {} | mask type: {}".format(img.shape, img.dtype, mask.shape, mask.dtype) )

def rotate(image):
    return ndimage.rotate(image,4, reshape=False)
         
def shift(image):
    return ndimage.shift(image,100.003, mode='nearest')

def rle(image):
    bytes = np.where(image.flatten()==1)[0]
    print(bytes)
    runs = []
    prev = -2
    for b in bytes:
        if(b>prev+1): runs.extend((b+1,0))
        runs[-1]+=1
        prev=b
    return ''.join([str(i) for i in runs])
rle(img)
#print(rle_mask(image_id))
