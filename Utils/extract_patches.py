import ConfigParser
import random

import numpy as np

# Load config params
config = ConfigParser.RawConfigParser()
config.read('configuration.txt')


# Load the original img and return the extracted patches
def get_img_training(img,
                     groundTruth,
                     patch_height,
                     patch_width,
                     N_pos_subimgs,
                     N_neg_subimgs,
                     inside_FOV):
    # Converting white pixels to 1 and black to 0
    groundTruth /= 255.

    img = img[:, 9:574, :]  # cut bottom and top so now it is 565*565
    groundTruth = groundTruth[:, 9:574, :]  # cut bottom and top so now it is 565*565

    data_consistency_check_img(img, groundTruth)

    # check masks are within 0-1
    assert (np.min(groundTruth) == 0 and np.max(groundTruth) == 1)

    print "\nimage/mask shape:"
    print img.shape
    print "image range (min-max): " + str(np.min(img)) + ' - ' + str(np.max(img))
    print "Ground Truth are within " + str(np.min(groundTruth)) + "-" + str(np.max(groundTruth)) + "\n"

    # extract the TRAINING patches from the full images
    patches_img, patches_groundTruth = extract_random(full_img=img, full_groundTruth=groundTruth, patch_h=patch_height,
                                                      patch_w=patch_width,
                                                      N_pos_patches=N_pos_subimgs, N_neg_patches=N_neg_subimgs,
                                                      inside=inside_FOV)

    data_consistency_check_patches(patches_img, patches_groundTruth)

    print "\ntrain PATCHES images/masks shape:"
    print patches_img.shape
    print "train PATCHES images range (min-max): " + str(np.min(patches_img)) + ' - ' + str(
        np.max(patches_img))

    return patches_img, patches_groundTruth  # , patches_imgs_test, patches_masks_test


# extract patches randomly in the full image
#  -- Inside OR in full image
def extract_random(full_img, full_groundTruth, patch_h, patch_w, N_pos_patches, N_neg_patches, inside=True):
    assert (len(full_img.shape) == 3 and len(full_groundTruth.shape) == 3)  # 3D arrays
    assert (full_img.shape[0] == 3)  # check the channel is 3
    assert (full_groundTruth.shape[0] == 1)  # masks only black and white
    assert (full_img.shape[1] == full_groundTruth.shape[1] and full_img.shape[2] == full_groundTruth.shape[2])

    # Total images would be N_pos_patches+N_neg_patches
    patches = np.empty(((N_pos_patches + N_neg_patches), full_img.shape[0], patch_h, patch_w))
    patches_groundTruth = np.empty(((N_pos_patches + N_neg_patches),))

    img_h = full_img.shape[1]  # height of the full image
    img_w = full_img.shape[2]  # width of the full image

    # (0,0) in the center of the image

    iter_tot = 0  # iter over the total number of patches (N_patches)
    k = 0
    count_pos_img = 0
    count_neg_img = 0
    while k < (N_pos_patches + N_neg_patches):
        # img_w - 1 to make sure if (565 - 1 - 13) is center element then 564 is 13th element
        x_center = random.randint(0 + int(patch_w / 2), (img_w - 1) - int(patch_w / 2))

        y_center = random.randint(0 + int(patch_h / 2), (img_h - 1) - int(patch_h / 2))

        if count_neg_img >= N_neg_patches and full_groundTruth[0, x_center, y_center] == 0:
            continue
        elif count_pos_img >= N_pos_patches and full_groundTruth[0, x_center, y_center] == 1:
            continue

        # check whether the patch is fully contained in the FOV
        if inside:
            if not is_patch_inside_FOV(x_center, y_center, img_w, img_h, patch_h):
                continue

        # y_center + int(patch_h / 2) + 1 so that total length before and after the center element is same
        patch = full_img[:, x_center - int(patch_w / 2):x_center + int(patch_w / 2) + 1,
                y_center - int(patch_h / 2):y_center + int(patch_h / 2) + 1]

        patch_groundTruth = full_groundTruth[0, x_center, y_center]

        if patch_groundTruth == 1:
            count_pos_img += 1
        elif patch_groundTruth == 0:
            count_neg_img += 1

        patches[iter_tot] = patch
        patches_groundTruth[iter_tot] = patch_groundTruth
        iter_tot += 1  # total
        k += 1  # per full_img

    print count_neg_img, count_pos_img
    return patches, patches_groundTruth


def data_consistency_check_img(img, groundTruth):
    assert (len(img.shape) == len(groundTruth.shape))
    assert (img.shape[1] == groundTruth.shape[1])
    assert (img.shape[2] == groundTruth.shape[2])
    assert (groundTruth.shape[0] == 1)
    assert (img.shape[0] == 3)


def data_consistency_check_patches(patches, patches_groundTruth):
    assert (len(patches.shape) == 4)
    assert (len(patches_groundTruth.shape) == 1)
    assert (patches.shape[0] == patches_groundTruth.shape[0])

    assert (patches.shape[2] == int(config.get('data attributes', 'patch_height')))
    assert (patches.shape[3] == int(config.get('data attributes', 'patch_width')))

    assert (patches.shape[1] == 3)


# check if the patch is fully contained in the FOV
def is_patch_inside_FOV(x, y, img_w, img_h, patch_h):
    x_ = x - int(img_w / 2)  # origin (0,0) shifted to image center
    y_ = y - int(img_h / 2)  # origin (0,0) shifted to image center
    R_inside = 270 - int(
        patch_h / 1.42)  # radius is 270 (from DRIVE db docs), minus the patch diagonal (assumed it is a square this
    # is the limit to contain the full patch in the FOV
    radius = np.sqrt((x_ * x_) + (y_ * y_))
    if radius < R_inside:
        return True
    else:
        return False
