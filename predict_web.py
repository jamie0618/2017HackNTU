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
model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'model/cnn_model.h5')

image_w = 160
image_h = 120

def ReadImage(image_w, image_h, image_name):
    im = Image.open(os.path.join(test_data_dir,image_name))
    im = im.resize((image_w, image_h), Image.BILINEAR )
    data = list(im.getdata())
    return data

def ReadFile(image_w,image_h):
    x_data = []
    y_data = []
    files = [f for f in os.listdir(test_data_dir)]
    for f in files:      
        y_data.append(int(f.split('.')[0][-1]))
        data = ReadImage(image_w,image_h,f)
        data = np.reshape(data,(image_w,image_h,3))
        x_data.append(data)
    return np.asarray(x_data), np.asarray(y_data)

if __name__ == "__main__":
    
    obike_classifier = load_model(model_path)
    x_data = ReadFile(image_name,image_w,image_h)
    x_data = x_data/255
    ans = obike_classifier.predict_classes(x_data)
    print("result :{0}".format(ans[0]))