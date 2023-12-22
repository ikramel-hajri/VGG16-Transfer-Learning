# VGG16 Transfer Learning

This repository explores Transfer Learning using the VGG16 model for image classification tasks, specifically focusing on recognizing flower types with the "tf_flowers" dataset.

## Overview

### What is Transfer Learning?

Transfer Learning is a machine learning technique where a pre-trained model is adapted for a new task. It leverages knowledge gained from a source task to improve learning on a target task.

### What is VGG16?

VGG16 is a deep convolutional neural network architecture designed for large-scale image recognition. It has 16 layers, including convolutional and fully connected layers. Pre-trained on ImageNet, it serves as an effective feature extractor for various computer vision tasks.

## Features

### Investigating Feature Maps with VGG16

1. Load the pre-trained VGG16 model using TensorFlow's Keras applications API and print its architecture.
2. Prepare an image for VGG16 using a provided function and predict the top-5 labels for a sample image.
3. Visualize feature maps of each convolutional layer and compare earlier layers to later layers.

### Transfer Learning with VGG16

1. Load and preprocess the "tf_flowers" dataset from TensorFlow datasets.
2. Modify the VGG16 architecture by adding a fully-connected classifier for flower type recognition.
3. Train the model by freezing the convolutional layers and training only the top dense layers.
4. Perform a round of fine-tuning by unfreezing the base model and training the entire model end-to-end with a low learning rate.
5. Report the obtained scores on the testing set.


## Resources

- [Keras API Documentation for VGG16](https://keras.io/api/applications/vgg/)
- [Transfer Learning Tutorial with Keras](https://keras.io/guides/transfer_learning/)

