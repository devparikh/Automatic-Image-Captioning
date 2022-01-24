# Importing all the libraries need for this project
import pandas as pd
import cv2
import random
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import os
import tensorflow as tf
import numpy as np

# Importing the images from Google Colab
image_set = "/content/Images"
image_dataset = []

image_size = 299
batch_size = 32
epochs = 20

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
  normalize_array = np.zeros((299,299))
  image = cv2.normalize(image, normalize_array, 0, 255, cv2.NORM_MINMAX)

  cv2.imshow(image)

  image_dataset.append(image)

# Importing the captions from a CSV file from Google Colab
caption_set = []

image_captions = pd.read_csv("/content/Images/captions.txt")
captions = image_captions["caption"]
print(captions.head())

for caption in captions:
  # Note that the since our captions are just used as a validation set by the model, we don't have to do any special preprocessing
  # We have to execute 2 main steps:
  # The first step is to convert our data from texts to sequences of numbers
  # The second step is to limit our sequences to a certain length, to perform this we will have to truncate some sequences while padding others

  # Define that Tokenizer that we are going to use to convert the texts to sequences of numbers
  tokenizer = Tokenizer(num_words = 50, char_level=False, oov_token="UNK")
  # Take the caption and fit it on this tokenizer
  tokenizer.fit_on_texts(caption)

  # Converting the text into sequences of numbers
  caption = tokenizer.texts_to_sequences(caption)
  # padding the inputs to make sure that they are all the same length when inputted into the model
  padded_captions = tf.keras.preprocessing.sequence.pad_sequences(caption, maxlen=50, dtype="int32", padding="post", truncating="post", value=0)

  # After preparing our validation captions for the training of the model, we will append all of the sequences of captions to caption_set
  caption_set.append(caption)

# Zipping together the 2 datasets 
image_captioning = list(zip(image_dataset, caption_set))
# Shuffling the zipped dataset because we don't want sets to be in orders when we split into training and testing data
random.shuffle(image_captioning)

# We are getting the split percentage for the training and testing sets 90% Training and 10% Testing
split_percentage = int(0.9*len(image_captioning))

# Splitting up the training and testing sets
train_image_captioning = image_captioning[:split_percentage]
validation_image_captioning = image_captioning[split_percentage:]

print(len(validation_image_captioning))
print(len(train_image_captioning))

# Unzipping the zipped training dataset into images and captions
unzipped_training_dataset = list(zip(*train_image_captioning))
unzipped_validation_dataset = list(zip(*validation_image_captioning))

training_images = []
training_captions = []

validation_images = []
validation_captions = []

def image_caption_separation(image_captioning_set, new_image_dataset, new_caption_dataset):
    # Iterating over the new list where position 1 contains a list of images and position 2 contains all of the captions
    for image in image_captioning_set[0]:
        new_image_dataset.append(image)

    for caption in image_captioning_set[1]:
        new_caption_dataset.append(caption)

image_caption_separation(unzipped_training_dataset, training_images, training_captions)
image_caption_separation(unzipped_validation_dataset, validation_images, validation_captions)

print(len(training_images))
print(len(training_captions))

# Converting the datasets to numpy arrays for the models
training_images = np.array(training_images)
validation_images = np.array(validation_images)

print(training_images.shape)
print(validation_images.shape)
