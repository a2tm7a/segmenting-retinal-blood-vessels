import ConfigParser

import h5py
from keras.models import Model
from keras.layers import Input, Convolution2D, MaxPooling2D, core, Dropout
from keras.preprocessing.image import ImageDataGenerator
import numpy as np
import sys

from keras.optimizers import SGD

np.random.seed(1573)

sys.path.append('../../segmenting-retinal-blood-vessels/')
from Utils.help_functions import load_train_data, load_val_data
from Utils.help_functions import print_confusion_matrix

# Load config params
config = ConfigParser.RawConfigParser()
config.read('../configuration.txt')


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

X_train, y_train = load_train_data()
X_val, y_val = load_val_data()

datagen_train = ImageDataGenerator(
    featurewise_center=False,  # set input mean to 0 over the dataset
    samplewise_center=False,  # set each sample mean to 0
    featurewise_std_normalization=False,  # divide inputs by std of the dataset
    samplewise_std_normalization=False,  # divide each input by its std
    zca_whitening=False,  # apply ZCA whitening
    rotation_range=0,  # randomly rotate images in the range (degrees, 0 to 180)
    width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
    height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
    horizontal_flip=True,  # randomly flip images
    vertical_flip=False)  # randomly flip images

datagen_train.fit(X_train)

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

    print lr, " learning rate"
    print iter_count, " iteration"
    model.compile(optimizer='sgd', loss='binary_crossentropy', metrics=['accuracy'])
    model.fit_generator(datagen_train.flow(X_train, y_train, batch_size=32), nb_epoch=1, verbose=1,
                        samples_per_epoch=X_train.shape[0])

    y_pred = model.predict_generator(datagen_train.flow(X_train, batch_size=32), val_samples=X_train.shape[0])
    print_confusion_matrix(y_pred, y_train)

    score = model.evaluate_generator(datagen_train.flow(X_val, y_val, batch_size=32), val_samples=X_val.shape[0])
    print score[1], score[0]
    val_accuracy = score[1]

    y_pred = model.predict_generator(datagen_train.flow(X_val, batch_size=32), val_samples=X_val.shape[0])
    print_confusion_matrix(y_pred, y_val)

    print val_accuracy, " - val accuracy"
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

del X_train
del y_train

del X_val
del y_val

model.save_weights('.' + str(config.get('data paths', 'saved_weights')) + "model.h5", overwrite=True)
