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

file_name = sys.argv[1]
model_path = 'model.h5'

base_dir = os.path.dirname(os.path.realpath(__file__))
test_data_dir = os.path.join(base_dir,'test_data')
image_w = 160
image_h = 120

def ReadImage(image_name, image_w, image_h):
    im = Image.open(os.path.join(test_data_dir,image_name))
    data = list(im.getdata())
    data = np.reshape(data,(image_w,image_h,3))
    return data

if __name__ == "__main__":
    
    obike_classifier = load_model(model_path)
    x_data = ReadImage(file_name,image_w,image_h)
    x_data = x_data/255
    ans = obike_classifier.predict_classes(x_data)
    print(ans)