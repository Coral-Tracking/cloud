# Import library
import os
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub

from PIL import Image
from io import BytesIO

from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array


# Flask Object 
app = Flask(__name__)

ALLOW_EXTENSION = {'jpg','jpeg'}

# Load the Machine Learning Model
model = tf.keras.models.load_model('model/classification_model.h5',custom_objects={'KerasLayer':hub.KerasLayer})

# Function to allow files format
def allow_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOW_EXTENSION

# Function to read the images
def read_image(image):
    img = Image.open(BytesIO(image))
    img = img.resize((150, 150), Image.ANTIALIAS)
    img = img_to_array(img)
    img /= 255
    img = np.expand_dims(img, axis=0)
    return img

# Server test function
@app.route('/') 
def index():
    return "Coral Tracking"

# Route Predict Images
@app.route('/predict', methods=['POST'])
def predict():
    image = request.files['image']

    if image and allow_file(image.filename):
        image = image.read()
        img = read_image(image)
        results = model.predict(img)
        max_value = np.max(results)

        if max_value <= 0.08:
            resp = jsonify({'message': 'Image can not be predicted'})
            resp.status_code = 400
            return resp 
        elif max_value > 0.08:
            result = np.argmax(results, axis = 1)

            if result == 0:
                Name = "Cyphastrea Microphthalma 042" +" ({:.0%})".format(max_value)
            elif result == 1:
                Name = "Pachyseris Speciosa 074" +" ({:.0%})".format(max_value)
            elif result == 2:
                Name = "Pachyseris Speciosa 056" +" ({:.0%})".format(max_value)
            elif result == 3:
                Name = "Acropora Clathrata 008" +" ({:.0%})".format(max_value)
            elif result == 4:
                Name = "Acropora Florida 198" +" ({:.0%})".format(max_value)
            elif result == 5:
                Name = "Cyphastrea Microphthalma 039" +" ({:.0%})".format(max_value)
            elif result == 6:
                Name = "Acropora Florida 027" +" ({:.0%})".format(max_value)
            elif result == 7:
                Name = "Acropora Clathrata 020" +" ({:.0%})".format(max_value)
            elif result == 8:
                Name = "Acropora Florida 001" +" ({:.0%})".format(max_value)
            elif result == 9:
                Name = "Diploastrea Heliopora 222" +" ({:.0%})".format(max_value)
            elif result == 10:
                Name = "Cyphastrea Microphthalma 076" +" ({:.0%})".format(max_value)
            elif result == 11:
                Name = "Acropora Florida 053" +" ({:.0%})".format(max_value)
            elif result == 12:
                Name = "Pachyseris Speciosa 068" +" ({:.0%})".format(max_value)
            elif result == 13:
                Name = "Acropora Clathrata 023" +" ({:.0%})".format(max_value)
            elif result == 14:
                Name = "Diploastrea Heliopora 253" +" ({:.0%})".format(max_value)
            elif result == 15:
                Name = "Diploastrea Heliopora 206" +" ({:.0%})".format(max_value)
            elif result == 16:
                Name = "Cyphastrea Microphthalma 078" +" ({:.0%})".format(max_value)
            elif result == 17:
                Name = "Acropora Florida 110" +" ({:.0%})".format(max_value)
            elif result == 18:
                Name = "Acropora Clathrata 076" +" ({:.0%})".format(max_value)
            elif result == 19:
                Name = "Acropora Clathrata 001" +" ({:.0%})".format(max_value)
            elif result == 20:
                Name = "Pachyseris Speciosa 069" +" ({:.0%})".format(max_value)
            elif result == 21:
                Name = "Pachyseris Speciosa 045" +" ({:.0%})".format(max_value)
            elif result == 22:
                Name = "Cyphastrea Microphthalma 048" +" ({:.0%})".format(max_value)
            elif result == 23:
                Name = "Diploastrea Heliopora 212" +" ({:.0%})".format(max_value)
            elif result == 24:
                Name = "Diploastrea Heliopora 261" +" ({:.0%})".format(max_value)
            return jsonify({'Name' : Name})
        else:
            res = jsonify({'message': 'Image extension is not allowed'})
            res.code_status = 400
            return res

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)



