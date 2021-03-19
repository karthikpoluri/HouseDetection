from __future__ import division, print_function
# coding=utf-8
import sys
import os
import glob
import re
import numpy as np
from tensorflow import keras
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import load_img



# Flask utils
from flask import Flask, redirect, url_for, request, render_template
from werkzeug.utils import secure_filename
from gevent.pywsgi import WSGIServer

# Define a flask app
app = Flask(__name__)

# loading model
model_path = 'House_detector.model02'
model = keras.models.load_model(model_path)
#model._make_predict_function()
print('Model loaded. Check http://127.0.0.1:5000/')

def predict_house(img_path,model):
    image = load_img(img_path, target_size=(224, 224))
    #image  = img_path
    img_array = np.expand_dims(image, axis=0)
    image = preprocess_input(img_array)
    prediction = np.round(model.predict(image))
    return prediction

@app.route('/', methods=['GET'])
def index():
    # Main page
    return render_template('index.html')

@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        # Get the file from post request
        f = request.files['file']

        # Save the file to ./uploads
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(
            basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)

        # Make prediction
        prediction = predict_house(file_path, model)

        if prediction[:, 0] == 1:
            text_House = "House"
        else:
            text_House = "Not House"
        return text_House

    return None


if __name__ == '__main__':
    app.run(debug=True)
