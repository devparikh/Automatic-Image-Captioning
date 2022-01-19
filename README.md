# Automatic-Image-Captioning
This is a project that uses a Attention Mechanism-based Encoder-Decoder Architecture for performing Image Captioning on the Flickr8k dataset on Kaggle.

![image](https://user-images.githubusercontent.com/47342287/150162642-28a7fd5b-3440-4db6-aa4e-30dfe8c13b33.png)

This is an image that represents the model architecture, there are 2 basic components in this architecture. We have the encoder which takes in the input image in the case of image captioning and the decoder is an LSTM(with attention layers) that decodes information from the feature maps that were created by the encoder and generates the most optimal words with a Beam Search Layer as the output of the decoder.
