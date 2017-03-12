import ConfigParser

import h5py
from keras.models import Model
from keras.layers import Input, merge, Convolution2D, MaxPooling2D, UpSampling2D, Reshape, core, Dropout
import numpy as np
import sys

from keras.optimizers import SGD
from keras.utils import np_utils

np.random.seed(1337)

sys.path.append('../../segmenting-retinal-blood-vessels/')
from Utils.help_functions import load_hdf5
from Utils.help_functions import print_confusion_matrix

# Load config params
config = ConfigParser.RawConfigParser()
config.read('../configuration.txt')

dataset_path = str(config.get('data paths', 'path_local'))
nb_classes = 2


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

    # up1 = merge([UpSampling2D(size=(2, 2))(conv3), conv2], mode='concat', concat_axis=1)
    conv4 = Convolution2D(64, 3, 3, activation='relu', border_mode='same')(conv3)
    conv4 = Dropout(0.2)(conv4)
    conv4 = Convolution2D(64, 3, 3, activation='relu', border_mode='same')(conv4)
    #
    # up2 = merge([UpSampling2D(size=(2, 2))(conv4), conv1], mode='concat', concat_axis=1)
    conv5 = Convolution2D(32, 3, 3, activation='relu', border_mode='same')(conv4)
    conv5 = Dropout(0.2)(conv5)
    conv5 = Convolution2D(32, 3, 3, activation='relu', border_mode='same')(conv5)
    #
    conv6 = Convolution2D(2, 1, 1, activation='relu', border_mode='same')(conv5)
    # conv6 = core.Reshape((2, patch_height * patch_width))(conv6)
    # conv6 = core.Permute((2, 1))(conv6)
    ############
    conv6 = core.Flatten()(conv6)
    conv7 = core.Dense(2)(conv6)
    conv7 = core.Activation('softmax')(conv7)

    model = Model(input=inputs, output=conv7)

    return model


model = get_unet(n_ch=int(config.get('data attributes', 'channels')),
                 patch_width=int(config.get('data attributes', 'patch_width')),
                 patch_height=int(config.get('data attributes', 'patch_height')))
model.summary()


def load_train_data():
    # Images from 21 to 38 are taken for training
    input_sequence = np.arange(21, 39)
    np.random.shuffle(input_sequence)

    j = 0
    X_train = None
    y_train = None
    # Testing purpose
    while j < len(input_sequence):
        print str(j) + " ", " data"

        temp_path1 = "." + dataset_path + "training_patches_" + str(input_sequence[j])
        temp_img1, temp_gt1 = load_hdf5(temp_path1)
        j += 1
        if X_train is None:
            X_train = temp_img1
            y_train = temp_gt1
        else:
            X_train = np.append(X_train, temp_img1, axis=0)
            y_train = np.append(y_train, temp_gt1, axis=0)

        del temp_img1
        del temp_gt1

    # TODO: Temp
    print y_train.dtype
    positive_examples = 0
    negative_examples = 0
    for i in range(y_train.shape[0]):
        if y_train[i] == 1:
            positive_examples += 1
        elif y_train[i] == 0:
            negative_examples += 1
        else:
            print "Something else"

    print positive_examples, negative_examples

    print "Before y_train", X_train.shape, y_train.shape
    y_train = np_utils.to_categorical(y_train, nb_classes)
    print "After y_train", y_train.shape

    # Shuffle data
    permutation = np.random.permutation(X_train.shape[0])
    print permutation
    X_train = X_train[permutation]
    y_train = y_train[permutation]
    return X_train, y_train


def load_val_data():
    # Testing on the left 2 images
    temp_path1 = "." + dataset_path + "training_patches_39"
    temp_path2 = "." + dataset_path + "training_patches_40"

    temp_img1, temp_gt1 = load_hdf5(temp_path1)
    temp_img2, temp_gt2 = load_hdf5(temp_path2)

    X_test = np.append(temp_img1, temp_img2, axis=0)
    y_test = np.append(temp_gt1, temp_gt2, axis=0)
    y_test = np_utils.to_categorical(y_test, nb_classes)
    return X_test, y_test


X_train, y_train = load_train_data()
X_val, y_val = load_val_data()

run_flag = True
weights = []
# Check if it is the first iteration
first_iter = True

# Setting the final accuracy to 0 just for the start
final_acc = 0.0
count_neg_iter = 0
iter_count = 1
nb_neg_cycles = 3

# TODO: find optimal learning rate
lr = 0.1

while run_flag:

    if first_iter:
        first_iter = False
    else:
        model.set_weights(np.asarray(weights))

    sgd = SGD(lr=lr)
    print iter_count," iteration"
    model.compile(optimizer='sgd', loss='binary_crossentropy', metrics=['accuracy'])
    model.fit(X_train, y_train, nb_epoch=1, verbose=1, validation_data=(X_val, y_val))

    y_pred = model.predict(X_train)
    print_confusion_matrix(y_pred, y_train)

    score = model.evaluate(X_val, y_val, verbose=1)
    print score[1], score[0]
    val_accuracy = score[1]

    y_pred = model.predict(X_val)
    print_confusion_matrix(y_pred, y_val)

    print val_accuracy," - val accuracy"
    print final_acc, " - final_accuracy" 
    if val_accuracy - final_acc > 0.0005:
        iter_count += 1
        # Update the weights if the accuracy is greater than .001
        weights = model.get_weights()
        print ("Updating the weights")
        # Updating the final accuracy
        final_acc = val_accuracy
        # Setting the count to 0 again so that the loop doesn't stop before reducing the learning rate n times
        # consecutively
        count_neg_iter = 0
    else:
        # If the difference is not greater than 0.005 reduce the learning rate
        lr /= 2.0
        print ("Reducing the learning rate by half")
        count_neg_iter += 1

        # If the learning rate is reduced consecutively for nb_neg_cycles times then the loop should stop
        if count_neg_iter > nb_neg_cycles:
            run_flag = False
            model.set_weights(np.asarray(weights))

            # Saving the model and weights in a separate file

del X_train
del y_train

del X_val
del y_val

model.save_weights('.' + str(config.get('data paths', 'saved_weights')) + "model.h5", overwrite=True)
