# -*- coding: utf-8 -*-
"""
Created on Sat Apr 22 14:46:12 2017

@author: jamie
"""

import os
import numpy as np
from PIL import Image
from keras.models import load_model

base_dir = os.path.dirname(os.path.realpath(__file__))
test_data_dir = os.path.join(base_dir,'test_data')
model_dir = os.path.join(base_dir,'model')
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

def Accuracy(y_real, y_predict):
    y_real = np.asarray(y_real)
    y_predict = np.asarray(y_predict)
    count = y_real - y_predict    
    return (len(y_real) - np.count_nonzero(count)) / len(y_real)
    
if __name__ == "__main__":
    
    model_path = os.path.join(model_dir,'cnn_model.h5')

    obike_classifier = load_model(model_path)
    obike_classifier.summary()
    x_data, y_data = ReadFile(image_w,image_h)
    x_data = x_data/255
    ans = obike_classifier.predict_classes(x_data)
    print("real label : {0}".format(y_data))
    print("predict ans: {0}".format(ans))
    
    acc = Accuracy(y_data, ans)
    print("Accuracy : {0}".format(acc))
    