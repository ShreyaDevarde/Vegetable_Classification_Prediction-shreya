# 🥕 Vegetable Image Classification using CNN

## 📌 Project Overview

This project is a deep learning–based web application that classifies vegetable images into predefined categories using a Convolutional Neural Network (CNN).

The trained CNN model is integrated into a Flask web application, allowing users to upload an image and receive real-time predictions.

---

## 🎯 Objective

The objective of this project is to:

- Build a CNN model for multi-class vegetable classification
- Apply image preprocessing and data augmentation
- Evaluate model performance on unseen test data
- Deploy the trained model using Flask
- Provide a user-friendly web interface for predictions

---

## 🧠 Model Architecture

The CNN architecture consists of:

- Convolutional Layers (Feature Extraction)
- MaxPooling Layers (Dimensionality Reduction)
- Flatten Layer
- Fully Connected (Dense) Layers
- Softmax Output Layer (15 Classes)

Loss Function: `Categorical Crossentropy`  
Optimizer: `Adam`  
Evaluation Metric: `Accuracy`

---

## 📂 Dataset Structure

Vegetable Images/
│
├── train/
├── validation/
└── test/

Each folder contains subfolders for individual vegetable classes.

Total Classes:
- Bean
- Bitter Gourd
- Bottle Gourd
- Brinjal
- Broccoli
- Cabbage
- Capsicum
- Carrot
- Cauliflower
- Cucumber
- Papaya
- Potato
- Pumpkin
- Radish
- Tomato
---
## 🔄 Project Workflow

1. Data Collection  
2. Data Preprocessing  
   - Resize images (150x150)
   - Normalize pixel values
   - Data augmentation
3. Model Building (CNN)
4. Model Training
5. Model Evaluation
6. Model Saving (`best_model.keras`)
7. Flask Deployment
8. Real-time Image Prediction
---
## 🌐 Web Application Flow

User → Upload Image → Flask Backend → CNN Model → Prediction Displayed

If confidence is below threshold:
> "Not a Vegetable" is displayed.
---
## 🛠️ Technologies Used

- Python
- TensorFlow / Keras
- NumPy
- Flask
- HTML
- CSS

---

## 🚀 How to Run the Project

### 1. Clone the Repository

```bash
git clone <your-repo-link>
cd Vegetables_Classification_Prediction-shreya

### 2.Create Virtual Environment (Recommended)
python -m venv venv
venv\Scripts\activate

### 3.Install Requirements
pip install -r requirements.txt

### 4. Run Flask App
python app.py

### 5.## 📊 Model Performance

### Accuracy Graph:

Test Loss: 0.40214115381240845
Test Accuracy: 0.8786666393280029

### 6.Open in browser:

http://127.0.0.1:5000

📊 Features

Image Upload

Real-time Prediction

Confidence-based validation

Clean UI

Multiple vegetable classes support.

