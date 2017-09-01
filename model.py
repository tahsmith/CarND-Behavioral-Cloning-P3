#!/usr/bin/env python

import os
from glob import glob

import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense, Flatten, Conv2D, InputLayer, Dropout, MaxPool2D, Cropping2D, Lambda
import cv2
from pickle import load

from sklearn.utils import shuffle


def build_model():
    image_shape = (85, 240, 1)
    model = Sequential()
    model.add(InputLayer(input_shape=image_shape))
    model.add(Lambda(lambda x: (x / 255.0) - 0.5))
    model.add(Conv2D(24, kernel_size=(5, 5), strides=(2, 2), activation='relu'))
    model.add(Conv2D(36, kernel_size=(5, 5), strides=(2, 2), activation='relu'))
    model.add(Conv2D(48, kernel_size=(5, 5), strides=(2, 2), activation='relu'))
    model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
    model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))

    model.add(Flatten())
    model.add(Dense(1200, activation='relu'))
    model.add(Dense(100, activation='sigmoid'))
    model.add(Dense(50, activation='sigmoid'))
    model.add(Dense(10, activation='sigmoid'))
    model.add(Dense(1))

    return model


def training_batches():
    paths = glob('data_cache/training-*.p')
    while True:
        for path in paths:
            with open(path, 'rb') as file:
                x, y = load(file)
                yield x, y


def validation_batches():
    paths = glob('data_cache/validation-*.p')
    while True:
        for path in paths:
            with open(path, 'rb') as file:
                x, y = load(file)
                yield x, y


def train(model):
    model.compile(loss='mse', optimizer='adam')
    model.fit_generator(training_batches(), epochs=1, steps_per_epoch=len(glob('data_cache/training-*.p')),
                        validation_data=validation_batches(),
                        validation_steps=len(glob('data_cache/validation-*.p')))
    model.save('model.h5')


def load_images(paths):
    images = np.empty([len(paths), 160, 320, 3], dtype=np.uint8)

    for i, path in enumerate(paths):
        image = cv2.imread(path)
        images[i, :, :, :] = image

    return images


def load_driving_log(filename, slice=slice(None)):
    prefix = os.path.split(filename)[0]

    def add_prefix(path):
        return os.path.join(prefix, path.strip())

    log = pd.DataFrame.from_csv(filename, header=0, index_col=None)
    left = log['left'][slice].transform(add_prefix)
    right = log['right'][slice].transform(add_prefix)
    center = log['center'][slice].transform(add_prefix)

    return left, center, right, np.array(log['steering'])[slice]


def flip(image):
    return image[:, :, ::-1, :]


def shift_to_pad(shift):
    return -shift if shift < 0 else 0, shift if shift >= 0 else 0


def shift_to_slice(shift, total_size):
    if shift >= 0:
        return slice(shift, shift + total_size)
    else:
        return slice(0, total_size)


def shift_image(image, shift):
    x_slice = shift_to_slice(shift[0], image.shape[1])
    y_slice = shift_to_slice(shift[1], image.shape[2])

    padded_image = np.pad(
        image,
        ((0, 0), shift_to_pad(shift[0]), shift_to_pad(shift[1]), (0, 0)),
        'constant',
        constant_values=128
    )

    sliced_image = padded_image[:, x_slice, y_slice, :]

    return sliced_image


def crop_image(img):
    return img[:, 50:135, 40:-40, :]


def preprocess(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).reshape((160, 320, 1))[50:135, 40:-40, :]


def preprocess_images(images):
    processed_images = np.empty([images.shape[0], 160, 320, 1])
    for i in range(images.shape[0]):
        processed_images[i, :, :, :] = cv2.cvtColor(images[i, :, :, :], cv2.COLOR_BGR2GRAY).reshape((160, 320, 1))
    return processed_images


def shift_img_rand(image, rand_range=5):
    shift = np.random.randint(-rand_range, rand_range + 1, (2,))
    return shift_image(image, shift)


def make_training_batch(left, center, right, steering):
    left, center, right = [preprocess_images(load_images(x)) for x in (left, center, right)]
    shifted_images = np.empty((0,) + center.shape[1:])
    shifts = [-40, 40]
    for shift in shifts:
        shifted_images = np.concatenate((shifted_images, shift_image(center, (0, shift))), axis=0)

    images = np.concatenate(
        list(map(
            crop_image,
            (center,
             left,
             right,
             shifted_images)
        )),
        axis=0)

    lr_cam_offset = 0.2

    shifted_steering = np.empty((0,))
    for shift in shifts:
        shifted_steering = np.concatenate((shifted_steering, steering + np.arctan(shift * -0.01)))

    steering = np.concatenate((
        steering,
        steering + lr_cam_offset,
        steering - lr_cam_offset,
        shifted_steering
    ))

    images = np.concatenate((images, flip(images)), axis=0)
    steering = np.concatenate((steering, -steering))

    return images, steering


def main():
    model = build_model()
    train(model)


if __name__ == '__main__':
    main()
