# Importing all the libraries for the model part of the project
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.layers import Embedding, LSTM, Dense, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.models import Sequential
import tensorflow as tf
import numpy as np
from preprocessing import *

# Passing the images through the pre-trained CNN
for image in training_images:
    image = tf.keras.application.inception_v3.preprocess_input(image)

# Initialzing the model
loading_model = tf.keras.applications.InceptionV3(include_top=False,
                                                weights='imagenet', 
                                                input_shape=(image_size, image_size,3))
input_layer = loading_model.input
output_layer = loading_model.layer[-1].output

# Creating the model
image_model = tf.keras.Model(input_layer, output_layer)
# Getting the output from the network
'''There is still a problem with batching that I have to solve here'''
feature_maps = image_model(training_images)

# Building LSTM:
model = Sequential()
# Starting the model with a embedding layer
model.add(Embedding(num_distinct_words, embedding_dim, input_length=50))

# Stacked LSTM layers
model.add(LSTM(10, activation="relu", return_sequences=True, return_state=True, dropout=0.2))
model.add(BatchNormalization(momentum=0.6))
model.add(LSTM(10, activation="relu", return_sequences=True, return_state=True, dropout=0.25))
model.add(LSTM(10, activation="relu", return_sequences=True, return_state=True, dropout=0.3))
model.add(BatchNormalization(momentum=0.7))
model.add(LSTM(10, activation="relu", return_sequences=True, dropout=0.4))

# This is a temporary layer, and once I learn how to implement Beam Search that layer will be used instead
model.add(Dense(num_distinct_words, activation="softmax"))

model.compile(optimizer=Adam(1e-3),
              loss=SparseCategoricalCrossentropy(),
              metrics=["accuracy"])

model.fit(training_images, training_captions, epochs=epochs, batch_size=batch_size, validation_data=[validation_images, validation_captions])
