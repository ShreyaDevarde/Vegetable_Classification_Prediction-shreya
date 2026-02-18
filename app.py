import base64
import os
import numpy as np
import tensorflow as tf
from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
from PIL import Image
from io import BytesIO

app = Flask(__name__)

# Upload folder
UPLOAD_FOLDER = "uploads"
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

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


@app.route('/result', methods=['GET', 'POST'])
def res():

    if request.form.get('captured_image'):

        # Get base64 image
        image_data = request.form['captured_image']
        image_data = image_data.split(',')[1]

        img_bytes = base64.b64decode(image_data)
        img = Image.open(BytesIO(img_bytes))
        img = img.resize((150, 150))

        filepath = "static/uploads/live_capture.png"
        img.save(filepath)

    else:
        f = request.files['image']
        filepath = os.path.join('static/uploads', f.filename)
        f.save(filepath)
        img = tf.keras.utils.load_img(filepath, target_size=(150, 150))

    img_arr = tf.keras.utils.img_to_array(img) / 255.0
    img_input = np.expand_dims(img_arr, axis=0)

    pred = np.argmax(model.predict(img_input))

    op = {
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

    result = op[pred]

    # Important for HTML display
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
