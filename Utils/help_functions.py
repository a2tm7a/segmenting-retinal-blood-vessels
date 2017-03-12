# prepare the mask in the right shape for the Unet
import ConfigParser

import h5py
import numpy as np
from keras.utils import np_utils
from sklearn.metrics import classification_report, confusion_matrix


def load_hdf5(infile):
    with h5py.File(infile, "r") as f:  # "with" close the file after its nested commands
        return f["img"][()], f["groundTruth"][()]


# y_pred and y_true are numpy arrays
def print_confusion_matrix(y_pred, y_true):
    y_pred = np.argmax(y_pred, axis=1)
    target_names = ['Class 0', 'Class 1']
    print classification_report(np.argmax(y_true, axis=1), y_pred, target_names=target_names)
    print (confusion_matrix(np.argmax(y_true, axis=1), y_pred))


nb_classes = 2
config = ConfigParser.RawConfigParser()
config.read('../configuration.txt')

dataset_path = str(config.get('data paths', 'path_local'))


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
