# -*- coding: utf-8 -*-
"""
Created on Sat Apr 22 14:46:12 2017

@author: jamie
"""

import os
import numpy as np
import sys
from PIL import Image
from keras.models import load_model

image_name = sys.argv[1]
model_path = 'model.h5'

image_w = 160
image_h = 120

def ReadImage(image_name):
    im = Image.open(image_name)
    return list(im.getdata())

def ReadFile(image_name,image_w,image_h):
    x_data = []    
    data = ReadImage(image_name)
    data = np.reshape(data,(image_w,image_h,3))
    x_data.append(data)
    return np.asarray(x_data)

if __name__ == "__main__":
    
    obike_classifier = load_model(model_path)
    x_data = ReadFile(image_name,image_w,image_h)
    x_data = x_data/255
    ans = obike_classifier.predict_classes(x_data)
    print("result :{0}".format(ans[0]))