import random

import numpy as np


# Load the original img and return the extracted patches
def get_img_training(img,
                     groundTruth,
                     patch_height,
                     patch_width,
                     N_subimgs,
                     inside_FOV):
    # train_imgs_original = load_hdf5(DRIVE_train_imgs_original)
    # train_masks = load_hdf5(DRIVE_train_groudTruth)  # masks always the same
    # # visualize(group_images(train_imgs_original[0:20,:,:,:],5),'imgs_train')#.show()  #check original imgs train


    # train_imgs = my_PreProc(train_imgs_original)

    # Converting white pixels to 1 and black to 0
    groundTruth = groundTruth / 255.

    img = img[:, 9:574, :]  # cut bottom and top so now it is 565*565
    groundTruth = groundTruth[:, 9:574, :]  # cut bottom and top so now it is 565*565

    data_consistency_check_img(img, groundTruth)

    # check masks are within 0-1
    assert (np.min(groundTruth) == 0 and np.max(groundTruth) == 1)

    print "\nimage/mask shape:"
    print img.shape
    print "image range (min-max): " + str(np.min(img)) + ' - ' + str(np.max(img))
    print "Ground Truth are within 0-1\n"

    # extract the TRAINING patches from the full images
    patches_img, patches_groundTruth = extract_random(img, groundTruth, patch_height, patch_width,
                                                      N_subimgs, inside_FOV)
    data_consistency_check_patches(patches_img, patches_groundTruth)

    print "\ntrain PATCHES images/masks shape:"
    print patches_img.shape
    print "train PATCHES images range (min-max): " + str(np.min(patches_img)) + ' - ' + str(
        np.max(patches_img))

    return patches_img, patches_groundTruth  # , patches_imgs_test, patches_masks_test


# extract patches randomly in the full image
#  -- Inside OR in full image
def extract_random(full_img, full_groundTruth, patch_h, patch_w, N_patches, inside=True):
    assert (len(full_img.shape) == 3 and len(full_groundTruth.shape) == 3)  # 3D arrays
    assert (full_img.shape[0] == 1 or full_img.shape[0] == 3)  # check the channel is 1 or 3
    assert (full_groundTruth.shape[0] == 1)  # masks only black and white
    assert (full_img.shape[1] == full_groundTruth.shape[1] and full_img.shape[2] == full_groundTruth.shape[2])

    patches = np.empty((N_patches, full_img.shape[0], patch_h, patch_w))
    patches_groundTruth = np.empty((N_patches, full_groundTruth.shape[0], patch_h, patch_w))

    img_h = full_img.shape[1]  # height of the full image
    img_w = full_img.shape[2]  # width of the full image

    # (0,0) in the center of the image

    iter_tot = 0  # iter over the total number of patches (N_patches)
    k = 0
    while k < N_patches:
        x_center = random.randint(0 + int(patch_w / 2), img_w - int(patch_w / 2))
        # print "x_center " +str(x_center)
        y_center = random.randint(0 + int(patch_h / 2), img_h - int(patch_h / 2))
        # print "y_center " +str(y_center)
        # check whether the patch is fully contained in the FOV
        if inside:
            if not is_patch_inside_FOV(x_center, y_center, img_w, img_h, patch_h):
                continue
        patch = full_img[:, y_center - int(patch_h / 2):y_center + int(patch_h / 2),
                x_center - int(patch_w / 2):x_center + int(patch_w / 2)]
        patch_groundTruth = full_groundTruth[:, y_center - int(patch_h / 2):y_center + int(patch_h / 2),
                            x_center - int(patch_w / 2):x_center + int(patch_w / 2)]
        patches[iter_tot] = patch
        patches_groundTruth[iter_tot] = patch_groundTruth
        iter_tot += 1  # total
        k += 1  # per full_img
    return patches, patches_groundTruth


def data_consistency_check_img(img, groundTruth):
    assert (len(img.shape) == len(groundTruth.shape))
    assert (img.shape[1] == groundTruth.shape[1])
    assert (img.shape[2] == groundTruth.shape[2])
    assert (groundTruth.shape[0] == 1)
    assert (img.shape[0] == 1 or img.shape[0] == 3)


def data_consistency_check_patches(patches, patches_groundTruth):
    assert (len(patches.shape) == len(patches_groundTruth.shape))
    assert (patches.shape[0] == patches_groundTruth.shape[0])
    assert (patches.shape[2] == patches_groundTruth.shape[2])
    assert (patches.shape[3] == patches_groundTruth.shape[3])
    assert (patches_groundTruth.shape[1] == 1)
    assert (patches.shape[1] == 1 or patches.shape[1] == 3)


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
