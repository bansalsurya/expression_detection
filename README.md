# Emotion-detection

## Introduction

This project aims to classify the emotion on a person's face into one of **seven categories**, using deep convolutional neural networks.
The model is trained on the **FER-2013** dataset which was published on International Conference on Machine Learning (ICML). 
This dataset consists of 35887 grayscale, 48x48 sized face images with **seven emotions** - angry, disgusted, fearful, happy, neutral, sad and surprised.

## Dependencies

* Python 3, [OpenCV 3 or 4](https://opencv.org/), [Tensorflow 1 or 2](https://www.tensorflow.org/)

## Usage (STEPS FOR EXECUTION)

* Download the FER-2013 dataset from [here](https://drive.google.com/file/d/1X60B-uR3NtqPd4oosdotpbDgy8KOfUdr/view?usp=sharing) and unzip it inside the folder and create a "data" Folder.

* If you want to train this model or train after making changes to the model, use `python emotions.py --mode train`.

* If you want to view the predictions without training again, then run "python emotions.py --mode display".

* The folder structure is of the form:  
  * data (folder)
  * "emotions.py" (file)
  * `haarcascade_frontalface_default.xml` (file)
  * `model.h5` (file)

* This implementation by default detects emotions on all faces in the webcam feed.

* With a simple 4-layer CNN, the test accuracy peaked at around 50 epochs at an accuracy of 63.2%.

## Algorithm

* First, we use **haar cascade** to detect faces in each frame of the webcam feed.

* The region of image containing the face is resized to **48x48** and is passed as input to the ConvNet.

* The network outputs a list of **softmax scores** for the seven classes.

* The emotion with maximum score is displayed on the screen.
