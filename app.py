import os
import numpy as np
import tensorflow as tf
from flask import Flask, render_template, request
from tensorflow.keras.models import load_model

app = Flask(__name__)

# Upload folder
UPLOAD_FOLDER = "uploads"
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Load trained model
model = load_model("vegetable_classification.h5", compile=False)

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
def result():
    if request.method == 'POST':
        file = request.files['image']
        
        if file:
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(filepath)

            # Preprocess image
            img = tf.keras.utils.load_img(filepath, target_size=(150,150))
            img_array = tf.keras.utils.img_to_array(img) / 255.0
            img_array = np.expand_dims(img_array, axis=0)

            # Prediction
            prediction = model.predict(img_array)
            pred_index = np.argmax(prediction)
            pred_label = class_labels[pred_index]

            return render_template('prediction.html', pred=pred_label, img_path=filepath)

    return render_template('prediction.html')

@app.route('/logout')
def logout():
    return render_template('logout.html')

if __name__ == '__main__':
    app.run(debug=True)
