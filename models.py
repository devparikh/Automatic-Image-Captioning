import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.layers import Embedding, LSTM, Dense, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.models import Sequential
import tensorflow as tf
import numpy as np

'''Importing VGG16 as our pre-trained CNN and passing our images through the model'''
# Loading in our model with the fully-connected output layer removed, weights based on imagenet, and input shape of (299,299,3)
loading_model = VGG16(include_top=False, weights="imagenet", input_shape=(image_size, image_size, 3))

# Input of the network
input = loading_model.input
# This is the output layer of this modified network, orignally this would be the last hidden layer
hidden_layers = loading_model.layers[-1].output
# Creating our model
image_feature_model = tf.keras.Model(input, hidden_layers)

batching = int(0.25*len(training_images))
half_batch = int(0.5*len(training_images))

training_batch_1 = training_images[:batching]
training_batch_2 = training_images[batching:half_batch]
training_batch_3 = training_images[half_batch:-batching]
training_batch_4 = training_images[-batching:]

# Making predictions on each of the batches
training_batch_1_maps = loading_model.predict(training_batch_1)
training_batch_2_maps = loading_model.predict(training_batch_2)
training_batch_3_maps = loading_model.predict(training_batch_3)
training_batch_4_maps = loading_model.predict(training_batch_4)

training_feature_maps = np.concatenate((training_batch_1_maps, training_batch_2_maps, training_batch_3_maps, training_batch_4_maps), axis=None)

# Deleting all of the batches to free up RAM
del training_batch_1
del training_batch_2
del training_batch_3
del training_batch_4

# Passing the validation set completely as the number of samples are not as large as the training dataset
feature_maps_validation = loading_model.predict(validation_images)
feature_maps_testing = loading_model.predict(testing_images)

for feature_map in feature_maps_validation:
  validation_feature_map_array = np.array(feature_map)
  validation_feature_maps = np.concatenate(validation_feature_map_array, axis=None)

for feature_map in feature_maps_testing:
  testing_feature_map_array = np.array(feature_map)
  testing_feature_maps = np.concatenate(testing_feature_map_array, axis=None)

print(training_feature_maps.shape)
print(validation_feature_maps.shape)
print(testing_feature_maps)

training_feature_maps = np.reshape(training_feature_maps, (6472,25088))
validation_feature_maps = np.reshape(validation_feature_maps, (1609,25088))

# Building LSTM:
model = Sequential()
# Starting the model with a embedding layer
model.add(Embedding(num_distinct_words, embedding_dim, input_length=25088))

# Stacked LSTM layers
model.add(LSTM(15, return_sequences=True, dropout=0.2))
model.add(BatchNormalization(momentum=0.6))
model.add(LSTM(15, return_sequences=True, dropout=0.25))
model.add(BatchNormalization(momentum=0.7))
model.add(LSTM(10, dropout=0.3))

model.add(Dense(num_distinct_words, activation="softmax"))

model.compile(optimizer=Adam(1e-3),
              loss=CategoricalCrossentropy(),
              metrics=["accuracy"])

model.fit(training_feature_maps, training_captions, epochs=epochs, batch_size=batch_size, validation_data=[validation_feature_maps, validation_captions])
