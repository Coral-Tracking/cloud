import os
from keras.models import load_model
from tensorflow import keras
from tensorflow.keras.preprocessing.image import img_to_array
import numpy as np

from PIL import Image
from io import BytesIO

from flask import Flask, request, jsonify

app = Flask(__name__)

ALLOW_EXTENSION = {'jpg','jpeg'}

class_names = ['Acropora Clathrata', 'Acropora Florida', 'Cyphastrea Microphthalma', 'Diploastrea Heliopora', 'Pachyseris Speciosa']
model = load_model('model/classification_model_fix_overfitting.h5')

model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001),loss='sparse_categorical_crossentropy',metrics=['accuracy'])


def allow_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOW_EXTENSION

def read_image(image):
    img = Image.open(BytesIO(image))
    img = img.resize((150, 150), Image.ANTIALIAS)
    x = img_to_array(img)
    x = np.expand_dims(img, axis=0)
    images = np.vstack([x])
    return images

def model_running(images):
    classes = model.predict(images, batch_size=10)
    prediction = model.predict(images)
    output_class = class_names[np.argmax(prediction)]
    percentage = 100 * np.max(prediction)
    return output_class, percentage

@app.route('/')
def index():
    return 'Coral Tracking'

@app.route('/predict', methods=['POST'])
def predict():
    image = request.files['image']

    if image and allow_file(image.filename):
        image = image.read()
        img = read_image(image)
        output_class, percentage = model_running(img)

        if percentage <= 0.08:
            resp = jsonify({'message': 'Image can not be predicted'})
            resp.status_code = 400
            return resp
        elif percentage > 0.08:
            Name = "{} ({:.2f}%)".format(output_class, percentage)
            resp = jsonify({'Name': Name})
            resp.status_code = 200
            return resp
        else:
            res = jsonify({'message': 'Image extension is not allowed'})
            res.code_status = 400
            return res

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)
