from PIL import Image

import h5py
import numpy as np
import os
from Utils.extract_patches import get_img_training


def write_hdf5(arr, outfile, name):
    with h5py.File(outfile, "w") as f:
        f.create_dataset(name, data=arr, dtype=arr.dtype)


# ------------Path of the images --------------------------------------------------------------
# train
original_imgs_train = "./DRIVE/training/images/"
groundTruth_imgs_train = "./DRIVE/training/1st_manual/"
borderMasks_imgs_train = "./DRIVE/training/mask/"
# test
original_imgs_test = "./DRIVE/test/images/"
groundTruth_imgs_test = "./DRIVE/test/1st_manual/"
borderMasks_imgs_test = "./DRIVE/test/mask/"
# ---------------------------------------------------------------------------------------------


Nimgs = 20
channels = 3
height = 584
width = 565
dataset_path = "./DRIVE_datasets_training_testing/"


def generate_datasets(imgs_dir, groundTruth_dir, borderMasks_dir, train_test="null"):
    img = np.empty((height, width, channels))
    groundTruth = np.empty((height, width))
    border_mask = np.empty((height, width))
    for path, subdirs, files in os.walk(imgs_dir):  # list all files, directories in the path
        for i in range(len(files)):

            # original
            print "original image: " + files[i]
            temp_img = Image.open(imgs_dir + files[i])
            print "size",temp_img.shape
            img = np.asarray(temp_img)

            # corresponding ground truth
            groundTruth_name = files[i][0:2] + "_manual1.gif"
            print "ground truth name: " + groundTruth_name
            temp_groundTruth = Image.open(groundTruth_dir + groundTruth_name)
            groundTruth = np.asarray(temp_groundTruth)

            # corresponding border masks
            border_masks_name = ""
            if train_test == "train":
                border_masks_name = files[i][0:2] + "_training_mask.gif"
            elif train_test == "test":
                border_masks_name = files[i][0:2] + "_test_mask.gif"
            else:
                print "specify if train or test!!"
                exit()
            print "border masks name: " + border_masks_name
            temp_border_mask = Image.open(borderMasks_dir + border_masks_name)
            border_mask = np.asarray(temp_border_mask)


generate_datasets(original_imgs_train, groundTruth_imgs_train,
                  borderMasks_imgs_train, "train")
