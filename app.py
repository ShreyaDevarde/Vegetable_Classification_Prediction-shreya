import os
import numpy as np
import tensorflow as tf
from flask import Flask, render_template, request
from tensorflow.keras.models import load_model

app = Flask(__name__)

# Upload folder
UPLOAD_FOLDER = "static/uploads"
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Ensure upload folder exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load trained model
model = load_model("best_model.keras", compile=False)

# Class labels (same order as training)
class_labels = {
    0: 'Bean',
    1: 'Bitter_Gourd',
    2: 'Bottle_Gourd',
    3: 'Brinjal',
    4: 'Broccoli',
    5: 'Cabbage',
    6: 'Capsicum',
    7: 'Carrot',
    8: 'Cauliflower',
    9: 'Cucumber',
    10: 'Papaya',
    11: 'Potato',
    12: 'Pumpkin',
    13: 'Radish',
    14: 'Tomato'
}

@app.route('/')
def home():
    return render_template('index.html')


@app.route('/prediction')
def prediction_page():
    return render_template('prediction.html')


@app.route('/result', methods=['POST'])
def res():

    f = request.files['image']
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], f.filename)
    f.save(filepath)

    img = tf.keras.utils.load_img(filepath, target_size=(150, 150))
    img_arr = tf.keras.utils.img_to_array(img) / 255.0
    img_input = np.expand_dims(img_arr, axis=0)

    # Prediction logic
    prediction = model.predict(img_input)
    confidence = np.max(prediction)
    pred = np.argmax(prediction)

    if confidence < 0.60:
        result = "Not a Vegetable"
    else:
        result = class_labels[pred]

    image_path = '/' + filepath

    return render_template(
        'logout.html',
        pred=result,
        image_path=image_path
    )


@app.route('/logout')
def logout():
    return render_template('logout.html')


if __name__ == '__main__':
    app.run(debug=True)
