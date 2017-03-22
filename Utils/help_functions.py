# prepare the mask in the right shape for the Unet
import ConfigParser

import h5py
import numpy as np
from keras.utils import np_utils


def load_hdf5(infile):
    with h5py.File(infile, "r") as f:  # "with" close the file after its nested commands
        return f["img"][()], f["groundTruth"][()]


def class_accuracy(y_predicted, y_value, threshold=0.5):
    # Calculates and returns the False Alarm Rate, False Reject Rate, True Alarm Rate, True Reject Rate.

    # Hypothesis
    false_reject = 0
    false_alarm = 0
    true_alarm = 0
    true_reject = 0

    # Total positive examples would be the sum of y_val because it would contain a 1 for every possible +ve example
    # and 0 for -ve example
    total_positive_examples = sum(y_value)
    total_negative_examples = len(y_value) - total_positive_examples

    for i in range(0, len(y_predicted)):
        # Checking for the hypothesis
        if y_predicted[i] >= threshold and y_value[i] == 0:
            false_alarm += 1
        elif y_predicted[i] < threshold and y_value[i] == 1:
            false_reject += 1
        elif y_predicted[i] >= threshold and y_value[i] == 1:
            true_alarm += 1
        elif y_predicted[i] < threshold and y_value[i] == 0:
            true_reject += 1

    print true_reject, false_alarm
    print false_reject, false_alarm

    return (false_alarm / float(total_negative_examples), false_reject / float(total_positive_examples),
            true_alarm / float(total_positive_examples), true_reject / float(total_negative_examples))


def load_train_data():
    nb_classes = 2
    config = ConfigParser.RawConfigParser()
    config.read('../configuration.txt')

    dataset_path = str(config.get('data paths', 'path_local'))

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
    nb_classes = 2
    config = ConfigParser.RawConfigParser()
    config.read('../configuration.txt')

    dataset_path = str(config.get('data paths', 'path_local'))

    # Testing on the left 2 images
    temp_path1 = "." + dataset_path + "training_patches_39"
    temp_path2 = "." + dataset_path + "training_patches_40"

    temp_img1, temp_gt1 = load_hdf5(temp_path1)
    temp_img2, temp_gt2 = load_hdf5(temp_path2)

    X_test = np.append(temp_img1, temp_img2, axis=0)
    y_test = np.append(temp_gt1, temp_gt2, axis=0)
    y_test = np_utils.to_categorical(y_test, nb_classes)
    return X_test, y_test


def rgb2gray(rgb):
    assert (len(rgb.shape) == 3)  # 3D Image
    assert (rgb.shape[0] == 3)
    bn_imgs = rgb[0, :, :] * 0.299 + rgb[1, :, :] * 0.587 + rgb[2, :, :] * 0.114
    bn_imgs = np.reshape(bn_imgs, (1, rgb.shape[1], rgb.shape[2]))
    return bn_imgs
