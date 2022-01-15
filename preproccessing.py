# Importing all the libraries need for this project
import pandas as pd
import glob
import cv2
import random
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras import layers
import re
from google.colab.patches import cv2_imshow
import os
import tensorflow as tf
import numpy as np

# Importing the images from Google Colab
image_set = "/content/Images"
image_dataset = []

image_size = 299
batch_size = 32

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

  # The guassian blur and normalization remove a small amount of noise, but with fastNiMeansDenoisingColored will denoise the image significantly more
  image = cv2.fastNlMeansDenoisingColored(image,None,10,10,7,21)

  cv2_imshow(image)

  image_dataset.append(image)

# Importing the captions from a CSV file from Google Colab
caption_set = []

image_captions = pd.read_csv("/content/Images/captions.txt")
captions = image_captions["caption"]
captions.head()

for caption in captions:
  # Note that the since our captions are just used as a validation set by the model, we don't have to do any special preprocessing
  # We have to execute 2 main steps:
  # The first step is to convert our data from texts to sequences of numbers
  # The second step is to limit our sequences to a certain length, to perform this we will have to truncate some sequences while padding others

  # Define that Tokenizer that we are going to use to convert the texts to sequences of numbers
  tokenizer = Tokenizer(num_words = 100, char_level=False, oov_token="UNK")
  # Take the caption and fit it on this tokenizer
  tokenizer.fit_on_texts(caption)

  # Converting the text into sequences of numbers
  caption = tokenizer.texts_to_sequences(caption)
  # padding the inputs to make sure that they are all the same length when inputted into the model
  padded_captions = tf.keras.preprocessing.sequence.pad_sequences(caption, maxlen=100, dtype="int32", padding="post", truncating="post", value=0)

  # After preparing our validation captions for the training of the model, we will append all of the sequences of captions to caption_set
  caption_set.append(caption)
