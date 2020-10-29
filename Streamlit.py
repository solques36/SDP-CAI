# -*- coding: utf-8 -*-
"""
Created on Thu Oct 29 17:34:10 2020

@author: Marcus
"""

import numpy as np
from tensorflow import keras
import matplotlib.pyplot as plt
import tensorflow.compat.v1 as tf
from tensorflow.keras.layers import Flatten, Dense, Conv2D, MaxPool2D, Dropout
from tensorflow.keras import Sequential, Model, layers 
from sklearn.metrics import confusion_matrix
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator , load_img , img_to_array 
from keras.layers.normalization import BatchNormalization
from keras.applications.vgg16 import preprocess_input
from tensorflow.keras.callbacks import Callback , EarlyStopping , ReduceLROnPlateau ,CSVLogger
from tensorflow.keras.callbacks import ModelCheckpoint
import itertools
import PIL
import os
from sklearn.utils import class_weight
from tensorflow.keras import regularizers
from pyexiv2 import Image
import streamlit as st
from load import predict



class_names_1 = ["ATCT","Apron","Arrival Immigration" ,"Boarding Gate", "Carkpark" ,"Departure Immigration", "Entrance" , "Fuel Farm" ]
  


st.title("Upload a file for classification")

uploaded_file = st.file_uploader("Choose an image...", type="jpg")

if uploaded_file is not None:
    new_image = image.load_img(uploaded_file)
    
    st.image(new_image, caption='Uploaded Image.', use_column_width=True)
    st.write("")
    st.write("Classifying...")
    prob , class_names , data = predict(uploaded_file, class_names_1)

    
    st.write(prob)
    st.write(class_names)
    st.write("Tags are below")
    st.write(data)

