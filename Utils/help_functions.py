# prepare the mask in the right shape for the Unet
import h5py
import numpy as np
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
