import ConfigParser

import h5py
from keras.models import Model
from keras.layers import Input, merge, Convolution2D, MaxPooling2D, UpSampling2D, Reshape, core, Dropout
import numpy as np
import sys


# TODO: Use function in helpfunction
# sys.path.append('../Utils')
# from help_functions import load_hdf5
# from Utils.help_functions import *


def load_hdf5(infile):
    with h5py.File(infile, "r") as f:  # "with" close the file after its nested commands
        return f["img"][()], f["groundTruth"][()]


# Load config params
config = ConfigParser.RawConfigParser()
config.read('../configuration.txt')

dataset_path = str(config.get('data paths', 'path_local'))


# Define the neural network
def get_unet(n_ch, patch_height, patch_width):
    inputs = Input((n_ch, patch_height, patch_width))
    conv1 = Convolution2D(32, 3, 3, activation='relu', border_mode='same')(inputs)
    conv1 = Dropout(0.2)(conv1)
    conv1 = Convolution2D(32, 3, 3, activation='relu', border_mode='same')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    #
    conv2 = Convolution2D(64, 3, 3, activation='relu', border_mode='same')(pool1)
    conv2 = Dropout(0.2)(conv2)
    conv2 = Convolution2D(64, 3, 3, activation='relu', border_mode='same')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    #
    conv3 = Convolution2D(128, 3, 3, activation='relu', border_mode='same')(pool2)
    conv3 = Dropout(0.2)(conv3)
    conv3 = Convolution2D(128, 3, 3, activation='relu', border_mode='same')(conv3)

    up1 = merge([UpSampling2D(size=(2, 2))(conv3), conv2], mode='concat', concat_axis=1)
    conv4 = Convolution2D(64, 3, 3, activation='relu', border_mode='same')(up1)
    conv4 = Dropout(0.2)(conv4)
    conv4 = Convolution2D(64, 3, 3, activation='relu', border_mode='same')(conv4)
    #
    up2 = merge([UpSampling2D(size=(2, 2))(conv4), conv1], mode='concat', concat_axis=1)
    conv5 = Convolution2D(32, 3, 3, activation='relu', border_mode='same')(up2)
    conv5 = Dropout(0.2)(conv5)
    conv5 = Convolution2D(32, 3, 3, activation='relu', border_mode='same')(conv5)
    #
    conv6 = Convolution2D(2, 1, 1, activation='relu', border_mode='same')(conv5)
    conv6 = core.Reshape((2, patch_height * patch_width))(conv6)
    conv6 = core.Permute((2, 1))(conv6)
    ############
    conv7 = core.Activation('softmax')(conv6)

    model = Model(input=inputs, output=conv7)

    # sgd = SGD(lr=0.01, decay=1e-6, momentum=0.3, nesterov=False)
    model.compile(optimizer='sgd', loss='categorical_crossentropy', metrics=['accuracy'])

    return model


# def trainmodel_epoch(train_sequence, val_sequence, model):
#     return final_training_accuracy, final_validation_accuracy


model = get_unet(n_ch=int(config.get('data attributes', 'channels')),
                 patch_width=int(config.get('data attributes', 'patch_width')),
                 patch_height=int(config.get('data attributes', 'patch_height')))

for i in range(5):
    input_sequence = np.arange(21, 39)
    np.random.shuffle(input_sequence)
    print input_sequence
    j = 0
    while j < len(input_sequence):
        print ("." + dataset_path)
        temp_path1 = "." + dataset_path + "training_patches_" + str(input_sequence[j])
        temp_path2 = "." + dataset_path + "training_patches_" + str(input_sequence[j + 1])
        temp_path3 = "." + dataset_path + "training_patches_" + str(input_sequence[j + 2])
        temp_img1, temp_gt1 = load_hdf5(temp_path1)
        temp_img2, temp_gt2 = load_hdf5(temp_path2)
        temp_img3, temp_gt3 = load_hdf5(temp_path3)
        j = j + 3
        X_train = np.append(temp_img1, temp_img2, axis=0)
        y_train = np.append(temp_gt1, temp_gt2, axis=0)

        del temp_img1
        del temp_gt1
        del temp_img2
        del temp_gt2

        X_train = np.append(X_train,temp_img3,axis=0)
        y_train = np.append(y_train,temp_gt3,axis=0)

        del temp_img3
        del temp_gt3

        print X_train.shape, y_train.shape
