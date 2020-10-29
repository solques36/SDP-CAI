# -*- coding: utf-8 -*-
"""
Created on Thu Oct 29 18:26:01 2020

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
import PIL
import os
from sklearn.utils import class_weight
from tensorflow.keras import regularizers
from pyexiv2 import Image
import streamlit as st









def predict( image1, class_names_1):
    
   new_image = image.load_img(image1 , target_size=(224,224))
   new_image = img_to_array(new_image)
   new_image = np.expand_dims(new_image, axis=0)
   new_image /=255.  
    
   model_1 = tf.keras.models.load_model('CAI Seed 42 Test 4 Group 1.h5')
    
   list_tag_names = []
   pred_1 = model_1.predict(new_image)  
   x = pred_1
  
   for index in range(len(pred_1[0])):
        
         if pred_1[0][index] > 0.2:
    
             accepted_index = index 
             tag_name = class_names_1[accepted_index]
             list_tag_names.append(tag_name)


   # if len(list_tag_names)==0:
   #      img = image.load_img(image1 , target_size=(224,224))
   #      data = img.read_exif()
   #      img.clear_exif()
   #      img.modify_exif({'Exif.Photo.UserComment': "Unknown"})
   #      data = img.read_exif()
   #      img.close()
    
   # else:
       
   #      img= image.load_img(image1 , target_size=(224,224))
   #      data = img.read_exif()
   #      img.clear_exif()
   #      img.modify_exif({'Exif.Photo.UserComment': list_tag_names})
   #      data = img.read_exif()
   #      img.close()
     
   return x , class_names_1 , list_tag_names