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
from flask import Flask, request
app = Flask(__name__)

image_name = sys.argv[1]
model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'model/cnn_model.h5')
obike_classifier = load_model(model_path)
image_w = 160
image_h = 120

def ReadImage(image_w, image_h, image_name):
    im = Image.open(image_name)
    im = im.resize((image_w, image_h), Image.BILINEAR )
    data = list(im.getdata())
    return data

def ReadFile(image_name, image_w,image_h):
    x_data = []
    data = ReadImage(image_w,image_h,image_name)
    data = np.reshape(data,(image_w,image_h,3))
    x_data.append(data)
    return np.asarray(x_data)

@app.route("/")
def run():
    image_name = request.args.get('input')
    x_data = ReadFile(image_name,image_w,image_h)
    x_data = x_data/255
    ans = obike_classifier.predict_classes(x_data)
    return "result :{0}\n".format(ans[0])


if __name__ == "__main__":

    obike_classifier = load_model(model_path)
    x_data = ReadFile(image_name,image_w,image_h)
    x_data = x_data/255
    ans = obike_classifier.predict_classes(x_data)
    print("result :{0}".format(ans[0]))
