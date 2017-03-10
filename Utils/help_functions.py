# TODO: Remove this and use one hot encoding
# prepare the mask in the right shape for the Unet
import h5py
import numpy as np


def masks_Unet(masks):
    assert (len(masks.shape) == 1)  # 4D arrays
    im_h = masks.shape[2]
    im_w = masks.shape[3]
    masks = np.reshape(masks, (masks.shape[0], im_h * im_w))
    new_masks = np.empty((masks.shape[0], im_h * im_w, 2))
    for i in range(masks.shape[0]):
        for j in range(im_h * im_w):
            if masks[i, j] == 0:
                new_masks[i, j, 0] = 1
                new_masks[i, j, 1] = 0
            else:
                new_masks[i, j, 0] = 0
                new_masks[i, j, 1] = 1
    return new_masks


def load_hdf5(infile):
    with h5py.File(infile, "r") as f:  # "with" close the file after its nested commands
        return f["img"][()], f["groundTruth"][()]
