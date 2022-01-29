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

'''Importing InceptionV3 as our pre-trained CNN and passing our images through the model'''

# Doing other preprocessing steps for the image before we send our image throught the network + Cache the feature maps from the network and store them
for image in training_images:
  # Preprocessing specific to InceptionV3
  image = tf.keras.applications.inception_v3.preprocess_input(image)

# Loading in our model with the fully-connected output layer removed, weights based on imagenet, and input shape of (299,299,3)
loading_model = tf.keras.applications.InceptionV3(include_top=False, weights="imagenet", input_shape=(image_size, image_size, 3))

# Input of the network
input = loading_model.input
# This is the output layer of this modified network, orignally this would be the last hidden layer
hidden_layers = loading_model.layers[-1].output
# Creating our model
image_feature_model = tf.keras.Model(input, hidden_layers)

concatenation = 0
for image in training_images:
  if concatenation <= len(training_images):
    training_set = training_images[concatenation] + training_images[concatenation+1]
    concatenation += 2

concatenation = 0
for images in validation_images:
  if concatenation <= len(validation_images):
    validation_set = validation_images[concatenation] + validation_images[concatenation+1]
    concatenation += 2

training_feature_maps = image_feature_model(training_set)
validation_feature_maps = image_feature_model(validation_set)

# Converting the training and validation feature maps to an array
training_feature_maps = np.array(training_feature_maps)
validation_feature_maps = np.array(validation_feature_maps)

print(training_feature_maps.shape())
print(validation_feature_maps.shape())
# Building LSTM:
model = Sequential()
# Starting the model with a embedding layer
model.add(Embedding(num_distinct_words, embedding_dim, input_length=50))

# Stacked LSTM layers
model.add(LSTM(15, return_sequences=True, dropout=0.2))
model.add(BatchNormalization(momentum=0.6))
model.add(LSTM(15, return_sequences=True, dropout=0.25))
model.add(BatchNormalization(momentum=0.7))
model.add(LSTM(10, dropout=0.3))

model.add(Dense(num_distinct_words, activation="linear"))
model.add(Dense(num_distinct_words, activation="softmax"))

model.add(tfa.seq2seq.BeamSearchDecoder(cell=keras.layers.Layer, beam_width=4, batch_size=64, length_penalty_weight=0.0, reorder_tensor_arrays=False))

model.compile(optimizer=Adam(1e-3),
              loss=SparseCategoricalCrossentropy(),
              metrics=["accuracy"])

model.fit(training_feature_maps, training_captions, epochs=epochs, batch_size=batch_size, validation_data=[validation_feature_maps, validation_captions])
