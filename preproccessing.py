# Importing all the libraries need for this project
import pandas as pd
import cv2
import random
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import os
import keras
import numpy as np
import tensorflow_hub as hub
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import SparseCategoricalCrossentropy
import tensorflow as tf
from google.colab.patches import cv2_imshow
import tensorflow_addons as tfa 
from tensorflow.keras.layers import Embedding, LSTM, Dense, BatchNormalization
from keras.applications.vgg16 import VGG16

# Importing the images from Google Colab
image_set = "/content/Images"
image_dataset = []


image_size = 224
batch_size = 64
epochs = 10 

num_distinct_words = 5000
embedding_dim = 10

for image in os.listdir(image_set):
  # Going to the image in the folder containing the image dataset and reading the file
  image = cv2.imread(os.path.join(image_set, image))
  # Resizing the image to the correct dimensions for InceptionV3
  image = cv2.resize(image, (image_size, image_size))

  '''Data Processing'''
  # We are applying Guassian Blurs because they allow us to remove noise from our images
  image = cv2.GaussianBlur(image, (5,5), 0, cv2.BORDER_DEFAULT)

  # we want to create this array where which the cv2 function on the next line will copy the original image and perform normalization
  normalize_array = np.zeros((224,224))
  image = cv2.normalize(image, normalize_array, 0, 255, cv2.NORM_MINMAX)

  #cv2.imshow(image)

  image_dataset.append(image)

# Importing the captions from a CSV file from Google Colab
caption_set = []

image_captions = pd.read_csv("/content/captions.txt")
captions = image_captions["caption"]
print(captions.head())

for caption in captions:
  # Note that the since our captions are just used as a validation set by the model, we don't have to do any special preprocessing
  # After preparing our validation captions for the training of the model, we will append all of the sequences of captions to caption_set
  caption_set.append(caption)

# We have to execute 2 main steps:
# The first step is to convert our data from texts to sequences of numbers
# The second step is to limit our sequences to a certain length, to perform this we will have to truncate some sequences while padding others
tokenizer = Tokenizer(num_words=num_distinct_words, char_level=False, oov_token="UNK")
# Take the caption and fit it on this tokenizer
tokenizer.fit_on_texts(caption_set)

# Converting the text into sequences of numbers
caption_set = tokenizer.texts_to_sequences(caption_set)
# padding the inputs to make sure that they are all the same length when inputted into the model
caption_set = tf.keras.preprocessing.sequence.pad_sequences(caption_set, maxlen=40, dtype="int32", padding="post", truncating="post", value=0)

# Zipping together the 2 datasets 
image_captioning = list(zip(image_dataset, caption_set))

# Shuffling the zipped dataset because we don't want sets to be in orders when we split into training and testing data
random.shuffle(image_captioning)

# We are getting the split percentage for the training and testing sets 90% Training and 10% Testing
split_percentage = int(0.8*len(image_captioning))

# Splitting up the training and testing sets
training_image_captioning = image_captioning[:split_percentage]
validation_image_captioning = image_captioning[split_percentage:]

#print(len(validation_image_captioning))
#print(len(training_image_captioning))

testing_image_captioning = []
testing = 0

for image_caption in validation_image_captioning[:10]:
  testing_image_captioning.append(image_caption)

validation_image_captioning = validation_image_captioning[10:]

# Unzipping the zipped training dataset into images and captions
unzipped_training_dataset = list(zip(*training_image_captioning))
unzipped_validation_dataset = list(zip(*validation_image_captioning))
unzipped_testing_dataset = list(zip(*testing_image_captioning))

training_images = []
validation_images = []
testing_images = []

training_captions = []
validation_captions = []
testing_captions = []

def image_caption_separation(image_captioning_set, new_image_dataset, new_caption_dataset):
    # Iterating over the new list where position 1 contains a list of images and position 2 contains all of the captions
    for image in image_captioning_set[0]:
      new_image_dataset.append(image)

    for caption in image_captioning_set[1]:
      new_caption_dataset.append(caption)

image_caption_separation(unzipped_training_dataset, training_images, training_captions)
image_caption_separation(unzipped_validation_dataset, validation_images, validation_captions)
image_caption_separation(unzipped_testing_dataset, testing_images, testing_captions)

# Problem:
# Figure out how to remove the image dataset and caption datset from the list that is stored in

training_images = np.array(training_images)
validation_images = np.array(validation_images)
testing_images = np.array(testing_images)

training_captions = np.array(training_captions)
validation_captions = np.array(validation_captions)
testing_captions = np.array(testing_captions)

print(training_images.shape)
print(validation_images.shape)
print(testing_images.shape)

print(training_captions.shape)
print(validation_captions.shape)
print(testing_captions.shape)
