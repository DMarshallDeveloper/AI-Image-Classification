#!/usr/bin/env python

"""Description:
The test.py is to evaluate your model on the test images.
***Please make sure this file work properly in your final submission***

©2018 Created by Yiming Peng and Bing Xue
"""
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array, ImageDataGenerator

# You need to install "imutils" lib by the following command:
#               pip install imutils
from imutils import paths
from sklearn.preprocessing import LabelBinarizer
import cv2
import os
import argparse

import numpy as np
import random
import tensorflow as tf

# Set random seeds to ensure the reproducible results
SEED = 309
np.random.seed(SEED)
random.seed(SEED)
# tf.set_random_seed(SEED)
tf.random.set_seed(SEED)
print(tf.version.VERSION)


def parse_args():
    """
    Pass arguments via command line
    :return: args: parsed args
    """
    # Parse the arguments, please do not change
    args = argparse.ArgumentParser()
    args.add_argument("--test_data_dir", default="data/test",
                      help="path to test_data_dir")
    args = vars(args.parse_args())
    return args


def load_images(test_data_dir, image_size, batch_size):
    classes = ["cherry", "strawberry", "tomato"]

    data = ImageDataGenerator(
        rescale=1. / 255)

    test_dataset = data.flow_from_directory(
        test_data_dir,  # this is the target directory
        target_size=image_size,
        color_mode='rgb',
        class_mode='categorical',
        batch_size = batch_size,
        classes=classes)

    return test_dataset


def evaluate(test_dataset):
    """
    Evaluation on test images
    ******Please do not change this function******
    :param test_dataset - the dataset
    :return: the accuracy
    """

    # Load Model
    model = load_model('model/model.h5')
    print(model.summary())
    return model.evaluate(test_dataset, verbose=1)


if __name__ == '__main__':
    # Parse the arguments
    args = parse_args()

    # Test folder
    test_data_dir = "data/test"

    # Image size, please define according to your settings when training your model.
    image_size = (64, 64)

    # batch size is 16 for evaluation
    batch_size = 16

    # Load images
    test_dataset = load_images(test_data_dir, image_size, batch_size)

    # Evaluation, please make sure that your training model uses "accuracy" as metrics, i.e., metrics=['accuracy']
    # loss, accuracy = evaluate(X_test, y_test)
    loss, accuracy = evaluate(test_dataset)
    print("loss={}, accuracy={}".format(loss, accuracy))
