#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 17 10:10:20 2019

@author: Manu
"""

from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['KMP_DUPLICATE_LIB_OK']='True'


# Initializing the CNN
model = Sequential()

# Step 1 - Convolution
model.add(Conv2D(32, (3, 3), input_shape=(64, 64, 3), activation='relu'))
# Step 2 - Pooling
model.add(MaxPooling2D(pool_size=(2,2)))
# Add another Convolutional Layer
model.add(Conv2D(32, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
# Step 3 - Flatenning
model.add(Flatten())
# Step 4 - Full Connection
model.add(Dense(units=128, activation='relu'))
model.add(Dense(units=1, activation='sigmoid'))
# Compiling CNN
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
# image preprocessing 
from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)

training_set = train_datagen.flow_from_directory(
        'dataset/training_set',
        target_size=(64, 64),
        batch_size=32,
        class_mode='binary')

test_set = test_datagen.flow_from_directory(
        'dataset/test_set',
        target_size=(64, 64),
        batch_size=32,
        class_mode='binary')


model.fit_generator(
        training_set,
        steps_per_epoch=8000,
        epochs=25,
        validation_data=test_set,
        validation_steps=2000)

import numpy as np
from keras.preprocessing import image

img = np.random.rand(64,64, 3)
img_path = 'dataset/single_prediction/cat_or_dog_1.jpg'
img = image.load_img(img_path, target_size=(64,64))
model.predict(img)