# -*- coding: utf-8 -*-
"""
Created on Sat Apr 22 14:45:44 2017

@author: jamie
"""
import os
import numpy as np
from PIL import Image
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense,Dropout,Flatten,Activation
from keras.layers import Convolution2D,MaxPooling2D
from keras.utils.np_utils import to_categorical

base_dir = os.path.dirname(os.path.realpath(__file__))
train_data_dir = os.path.join(base_dir,'train_data')
model_dir = os.path.join(base_dir,'model')
image_w = 160
image_h = 120

def ReadImage(image_w, image_h, image_name):
    im = Image.open(os.path.join(train_data_dir,image_name))
    im = im.resize((image_w, image_h), Image.BILINEAR )
    data = list(im.getdata())
    return data

def ReadFold(image_w,image_h):
    x_data = []
    y_data = []
    files = [f for f in os.listdir(train_data_dir)]
    for f in files:       
        y_data.append(int(f.split('.')[0][-1]))
        data = ReadImage(image_w,image_h,f)
        data = np.reshape(data,(image_w,image_h,3))
        x_data.append(data)
    y_data = to_categorical(np.asarray(y_data,dtype=np.int32))  
    return np.asarray(x_data), np.asarray(y_data)
      
def build_model(image_w, image_h):
    model = Sequential()

    # CNN part 
    model.add(Convolution2D(32,(3,3),padding='valid',input_shape=(image_w,image_h,3)))       
    model.add(Activation('relu'))  
    model.add(Convolution2D(32,(3,3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.3))
    
    model.add(Convolution2D(64,(3,3)))
    model.add(Activation('relu'))  
    model.add(Convolution2D(64,(3,3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.3))
    
    model.add(Convolution2D(64,(3,3)))
    model.add(Activation('relu'))  
    model.add(Convolution2D(64,(3,3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.3))
            
    # Fully connected part
    model.add(Flatten())     
    model.add(Dense(128))
    model.add(Activation('relu'))
    model.add(Dropout(0.25))        
        
    model.add(Dense(2))
    model.add(Activation('softmax'))
    model.compile(loss='categorical_crossentropy',
                  optimizer='Adam',
                  metrics=['accuracy'])
    model.summary() 
    return model

if __name__ == "__main__":
    epoch = 10
    batch = 10
    
    # build model
    print("build model...")
    obike_classifier = build_model(image_w, image_h)
    
    # read image
    print("read image...")
    x,y = ReadFold(image_w,image_h)
       
    # use image data generator
    train_datagen = ImageDataGenerator(rescale=1./255
                                       ,shear_range=0.2
                                       ,zoom_range=0.2
                                       ,horizontal_flip=True)
    test_datagen = ImageDataGenerator(rescale=1./255)
    
    train_generator = train_datagen.flow(x,y,batch_size=batch)
    validation_generator = test_datagen.flow(x,y,batch_size=batch)
 
    # fit model
    print("fit model...")
    obike_classifier.fit_generator(train_generator
                                     ,steps_per_epoch=30
                                     ,epochs=epoch
                                     ,validation_data=validation_generator
                                     ,validation_steps=20)  
    
    obike_classifier.save(os.path.join(model_dir,'cnn_model.h5'))
