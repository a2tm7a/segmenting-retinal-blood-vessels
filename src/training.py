import ConfigParser

import h5py
from keras.models import Model
from keras.layers import Input, merge, Convolution2D, MaxPooling2D, UpSampling2D, Reshape, core, Dropout
import numpy as np
import sys
from sklearn.metrics import classification_report, confusion_matrix

np.random.seed(1234)

sys.path.append('../../segmenting-retinal-blood-vessels/')
from Utils.help_functions import load_hdf5, masks_Unet

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


model = get_unet(n_ch=int(config.get('data attributes', 'channels')),
                 patch_width=int(config.get('data attributes', 'patch_width')),
                 patch_height=int(config.get('data attributes', 'patch_height')))
model.summary()

for i in range(1):
    # Images from 21 to 38 are taken for training
    input_sequence = np.arange(21, 39)
    np.random.shuffle(input_sequence)
    print '\n' + str(input_sequence), str(i) + "th iteration"

    j = 0

    # Testing purpose
    # while j < len(input_sequence):
    while j < 1:
        print str(j) + " ", str(j + 1) + " ", str(j + 2) + " -> ", input_sequence[j], input_sequence[j + 1], \
            input_sequence[j + 2]

        temp_path1 = "." + dataset_path + "training_patches_" + str(input_sequence[j])
        temp_path2 = "." + dataset_path + "training_patches_" + str(input_sequence[j + 1])
        temp_path3 = "." + dataset_path + "training_patches_" + str(input_sequence[j + 2])
        temp_img1, temp_gt1 = load_hdf5(temp_path1)
        temp_img2, temp_gt2 = load_hdf5(temp_path2)
        temp_img3, temp_gt3 = load_hdf5(temp_path3)
        j += 3
        X_train = np.append(temp_img1, temp_img2, axis=0)
        y_train = np.append(temp_gt1, temp_gt2, axis=0)

        del temp_img1
        del temp_gt1
        del temp_img2
        del temp_gt2

        X_train = np.append(X_train, temp_img3, axis=0)
        y_train = np.append(y_train, temp_gt3, axis=0)

        del temp_img3
        del temp_gt3

        print X_train.shape, y_train.shape
        y_train = masks_Unet(y_train)
        print y_train.shape

        # Shuffle data
        permutation = np.random.permutation(X_train.shape[0])
        X_train = X_train[permutation]
        y_train = y_train[permutation]

        model.fit(X_train, y_train, nb_epoch=1, validation_split=.1, verbose=1)

        del X_train
        del y_train

model.save_weights('..' + str(config.get('data paths', 'saved_weights')) + "model.h5", overwrite=True)

# Testing on the left 2 images
temp_path1 = "." + dataset_path + "training_patches_39"
temp_path2 = "." + dataset_path + "training_patches_40"

temp_img1, temp_gt1 = load_hdf5(temp_path1)
temp_img2, temp_gt2 = load_hdf5(temp_path2)

X_test = np.append(temp_img1, temp_img2, axis=0)
y_test = np.append(temp_gt1, temp_gt2, axis=0)
y_test = masks_Unet(y_test)

del temp_img1
del temp_gt1
del temp_img2
del temp_gt2

print "Validation Data"
score = model.evaluate(X_test, y_test, verbose=1)
print score[1], score[0]

y_pred = model.predict(X_test)
y_pred = np.argmax(y_pred, axis=1)

target_names = ['Class 0', 'Class 1']
print y_test, y_pred
print classification_report(np.argmax(y_test, axis=1), y_pred, target_names=target_names)
print (confusion_matrix(np.argmax(y_test, axis=1), y_pred))
