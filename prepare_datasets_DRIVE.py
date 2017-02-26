import ConfigParser
from PIL import Image

import h5py
import numpy as np
import os
from Utils.extract_patches import get_img_training


def write_hdf5(img, outfile, groundTruth):
    with h5py.File(outfile, "w") as f:
        f.create_dataset("img", data=img, dtype=img.dtype)
        f.create_dataset("groundTruth", data=groundTruth, dtype=groundTruth.dtype)


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

# Load config params
config = ConfigParser.RawConfigParser()
config.read('configuration.txt')

dataset_path = config.get('data paths', 'path_local')


def swapaxes_img(img):
    temp = np.swapaxes(img, 0, 2)
    temp = np.swapaxes(temp, 1, 2)
    return temp


def generate_datasets(imgs_dir, groundTruth_dir, borderMasks_dir, train_test="null"):
    for path, subdirs, files in os.walk(imgs_dir):  # list all files, directories in the path
        for i in range(len(files)):

            # original image
            print "original image: " + files[i]
            temp_img = Image.open(imgs_dir + files[i])
            # To reshape the image as (channels x height x width)
            temp_img = swapaxes_img(temp_img)
            img = np.asarray(temp_img)
            print img.shape

            # corresponding ground truth
            groundTruth_name = files[i][0:2] + "_manual1.gif"
            print "ground truth name: " + groundTruth_name
            temp_groundTruth = Image.open(groundTruth_dir + groundTruth_name)
            # To reshape the ground truth as (channels x height x width)
            temp_groundTruth = np.reshape(temp_groundTruth, (
                1, np.asarray(temp_groundTruth).shape[0], np.asarray(temp_groundTruth).shape[1]))
            groundTruth = temp_groundTruth
            print groundTruth.shape

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
            temp_border_mask = np.reshape(temp_border_mask, (
                1, np.asarray(temp_border_mask).shape[0], np.asarray(temp_border_mask).shape[1]))
            border_mask = temp_border_mask
            print border_mask.shape

            # TODO: Preprocessing  of image

            patches_img, patches_groundTruth = get_img_training(img=img, groundTruth=groundTruth,
                                                                patch_height=int(
                                                                    config.get('data attributes', 'patch_height')),
                                                                patch_width=int(
                                                                    config.get('data attributes', 'patch_width')),
                                                                N_subimgs=int(
                                                                    config.get('training settings', 'N_subimgs')),
                                                                inside_FOV=config.getboolean('training settings',
                                                                                             'inside_FOV'))

            write_hdf5(img=patches_img, outfile=dataset_path + "training_patches_" + files[i][0:2],
                       groundTruth=patches_groundTruth)


if not os.path.exists(dataset_path):
    os.makedirs(dataset_path)

generate_datasets(original_imgs_train, groundTruth_imgs_train,
                  borderMasks_imgs_train, "train")
