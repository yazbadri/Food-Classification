## In this Repo:

- Part_01: Datasets
- Part_02: Notebooks of iterations, model files, readme.md
- Part_03: 
- Part_04: 
- Part_05: 

# Problem Statement

In this project, we will be creating a neural network model to try to correctly classify food images. 

## Datasets used:

- Food-5k: https://www.kaggle.com/binhminhs10/food5k

- Food-11: https://www.epfl.ch/labs/mmspg/downloads/food-image-datasets/

## Libraries used:

import pandas as pd
import numpy as np
from numpy.random import seed

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import Model
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (Dense, Dropout, Flatten, Conv2D)
from tensorflow.keras.layers import SeparableConv2D, ReLU, MaxPooling2D, Add, Input
from tensorflow.keras.layers import BatchNormalization, MaxPool2D, GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam, SGD, RMSprop
from tensorflow.keras.preprocessing import image
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.applications.vgg16 import decode_predictions, preprocess_input
from tensorflow.keras.applications.xception import Xception
from tensorflow.keras.applications.xception import decode_predictions, preprocess_input


from PIL import Image

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
%matplotlib inline
import seaborn as sns
import requests
import os


## Metrics:

We will be measuring accuracy of the model predictions.
Goal: 90%

## Steps:

- EDA & Preprocessing
- NN Model on Food-5k data (Food, Not-food)
- NN Models on subset of Food-11 data (food classes - 101 total, starting with subset 3 classes)
- NN Model on larger subset
- NN model on whole Food-11 dataset
- Summary, Conclusions & Recommendations

## Findings:

- Food-5k got to 100% accuracy with CNN model
- Food-11 more difficult
- Using pre-trained models helped boost accuracy

