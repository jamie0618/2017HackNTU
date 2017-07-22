# -*- coding: utf-8 -*-
"""
Created on Sat Apr 22 14:45:44 2017

@author: jamie
"""
import os
import numpy as np
from PIL import Image
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Dense
from keras.layers import GlobalAveragePooling2D
from keras.optimizers import SGD
from keras.utils.np_utils import to_categorical

from keras.applications.inception_v3 import InceptionV3
from keras.models import Model

base_dir = os.path.dirname(os.path.realpath(__file__))
data_dir = os.path.join(base_dir,'data')
model_dir = os.path.join(base_dir,'model')
image_w = 160
image_h = 120

def ReadImage(image_name):
    im = Image.open(os.path.join(data_dir,image_name))
    return list(im.getdata())

def ReadFile(image_w,image_h):
    x_data = []
    y_data = []
    files = [f for f in os.listdir(data_dir)]
    num_image = len(files)
    for f in files:       
        y_data.append(int(f.split('_resized')[0][-1]))
        data = ReadImage(f)
        data = np.reshape(data,(image_w,image_h,3))
        x_data.append(data)
    y_data = to_categorical(np.asarray(y_data,dtype=np.int32))  
    return np.asarray(x_data), np.asarray(y_data), num_image

if __name__ == "__main__":
    epoch = 10
    batch = 10
    
    print("Read Data...")      
    x,y,n = ReadFile(image_w,image_h)
    valid = int(0.2*n)
    x_train = x[:]
    y_train = y[:]
    x_valid = x[:]
    y_valid = y[:]
       
    # use image data generator    
    train_datagen = ImageDataGenerator(rescale=1./255
                                       ,shear_range=0.2
                                       ,zoom_range=0.2
                                       ,horizontal_flip=True)
    test_datagen = ImageDataGenerator(rescale=1./255)
    
    train_generator = train_datagen.flow(x_train,y_train,batch_size=batch)
    validation_generator = test_datagen.flow(x_valid,y_valid,batch_size=batch)   
    
    print("Create base model...")
    # create the base pre-trained model
    base_model = InceptionV3(weights='imagenet', include_top=False)
    
    # add a global spatial average pooling layer
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    # let's add a fully-connected layer
    x = Dense(512, activation='relu')(x)
    # and a logistic layer -- let's say we have 200 classes
    predictions = Dense(2, activation='softmax')(x)
    
    # this is the model we will train
    model = Model(inputs=base_model.input, outputs=predictions)
    
    # first: train only the top layers (which were randomly initialized)
    # i.e. freeze all convolutional InceptionV3 layers
    for layer in base_model.layers:
        layer.trainable = False
    
    # compile the model (should be done *after* setting layers to non-trainable)
    model.compile(optimizer='Adam', loss='categorical_crossentropy',metrics=['accuracy'])
    
    print("Fit base model...")
    # train the model on the new data for a few epochs
    model.fit_generator(train_generator
                        ,steps_per_epoch=30
                        ,epochs=epoch
                        ,validation_data=validation_generator
                        ,validation_steps=20)
    
    # at this point, the top layers are well trained and we can start fine-tuning
    # convolutional layers from inception V3. We will freeze the bottom N layers
    # and train the remaining top layers.
    
    # let's visualize layer names and layer indices to see how many layers
    # we should freeze:
    for i, layer in enumerate(base_model.layers):
       print(i, layer.name)
    
    # we chose to train the top 2 inception blocks, i.e. we will freeze
    # the first 249 layers and unfreeze the rest:
    for layer in model.layers[:249]:
       layer.trainable = False
    for layer in model.layers[249:]:
       layer.trainable = True
    
    # we need to recompile the model for these modifications to take effect
    # we use SGD with a low learning rate
    model.compile(optimizer=SGD(lr=0.001, momentum=0.9), loss='categorical_crossentropy')
    
    print("Fit final model...")
    # we train our model again (this time fine-tuning the top 2 inception blocks
    # alongside the top Dense layers
    model.fit_generator(train_generator
                        ,steps_per_epoch=30
                        ,epochs=epoch
                        ,validation_data=validation_generator
                        ,validation_steps=20)
    model.save('model.h5')