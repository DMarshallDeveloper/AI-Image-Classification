#!/usr/bin/env python

"""Description:
The train.py is to build your CNN model, train the model, and save it for later evaluation(marking)
This is just a simple template, you feel free to change it according to your own style.
However, you must make sure:
1. Your own model is saved to the directory "model" and named as "model.h5"
2. The "test.py" must work properly with your model, this will be used by tutors for marking.
3. If you have added any extra pre-processing steps, please make sure you also implement them in "test.py" so that they can later be applied to test images.

©2018 Created by Yiming Peng and Bing Xue
"""
import datetime
import random
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.layers import Dropout, Flatten, Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Set random seeds to ensure the reproducible results
SEED = 309
directory = "data/train"
np.random.seed(SEED)
random.seed(SEED)
# tf.set_random_seed(SEED)
tf.random.set_seed(SEED)


def construct_model():
    """
    Construct the CNN model.
    ***
        Please add your model implementation here, and don't forget compile the model
        E.g., model.compile(loss='categorical_crossentropy',
                            optimizer='sgd',
                            metrics=['accuracy'])
        NOTE, You must include 'accuracy' in as one of your metrics, which will be used for marking later.
    ***
    :return: model: the initial CNN model
    """
    model = Sequential()

    model.add(Conv2D(64, (3, 3), input_shape=(64, 64, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())  # this converts our 3D feature maps to 1D feature vectors

    model.add(Dense(units=64, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(units=64, activation='relu'))
    model.add(Dropout(0.5))

    model.add(Dense(units=3, activation='softmax'))
    model.compile(loss='categorical_crossentropy',
                  optimizer='Adamax',
                  metrics=['accuracy'])
    model.summary()
    return model


def train_model(model):
    """
    Train the CNN model
    ***
        Please add your training implementation here, including pre-processing and training
    ***
    :param model: the initial CNN model
    :return:model:   the trained CNN model
    """

    batch_size = 32

    classes = ["cherry", "strawberry", "tomato"]

    data = ImageDataGenerator(
        rescale=1. / 255,
        shear_range=0.2,
        rotation_range=40,
        zoom_range=0.2,
        horizontal_flip=True,
        vertical_flip=True,
        validation_split=0.1)

    train = data.flow_from_directory(
        "data/train",  # this is the target directory
        target_size=(64, 64),
        color_mode='rgb',
        class_mode='categorical',
        classes=classes,
        subset='training')

    validation = data.flow_from_directory(
        "data/train",  # this is the target directory
        target_size=(64, 64),
        color_mode='rgb',
        class_mode='categorical',
        classes=classes,
        subset='validation')


    start_time = datetime.datetime.now()  # Track learning starting time
    history = model.fit(train, validation_data=validation, epochs=60, verbose=1, callbacks=[EarlyStopping(
        monitor='val_accuracy', patience=8, mode='max', restore_best_weights=True)])

    end_time = datetime.datetime.now()  # Track learning ending time
    execution_time = (end_time - start_time).total_seconds()  # Track execution time
    print("Learn: execution time={t:.3f} seconds".format(t=execution_time))
        # Get training and test loss histories
    training_accuracy = history.history['accuracy']
    test_accuracy = history.history['val_accuracy']

    # Create count of the number of epochs
    epoch_count = range(1, len(training_accuracy) + 1)

    # Visualize loss history
    plt.plot(epoch_count, training_accuracy, 'r--')
    plt.plot(epoch_count, test_accuracy, 'b-')
    plt.legend(['Training Accuracy', 'Test Accuracy'])
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Test accuracy for the CNN')
    plt.show();
    return model


def save_model(model):
    """
    Save the keras model for later evaluation
    :param model: the trained CNN model
    :return:
    """
    # ***
    #   Please remove the comment to enable model save.
    #   However, it will overwrite the baseline model we provided.
    # ***
    model.save("model/model.h5")
    print("Model Saved Successfully.")


if __name__ == '__main__':
    physical_devices = tf.config.experimental.list_physical_devices('GPU')
    tf.config.experimental.set_memory_growth(physical_devices[0], True)

    model = construct_model()
    model = train_model(model)
    save_model(model)
