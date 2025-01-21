import os
import numpy as np
import cv2
import joblib
import tensorflow as tf
from flask import Flask, render_template, request, jsonify

# Initialize Flask app
app = Flask(__name__)

# Paths to model and encoder
cnn_model_path = 'models/cnn.tflite'  # Changed to .tflite
rfc_model_path = 'models/rfc2.pkl'
label_encoder_path = 'models/label_encoder.pkl'

# Load the models and encoder
# Load TensorFlow Lite model
interpreter = tf.lite.Interpreter(model_path=cnn_model_path)
interpreter.allocate_tensors()
print("CNN model loaded successfully.")

# Load Random Forest Classifier
rf_classifier = joblib.load('models/rfc2.pkl')
print("Random Forest model loaded successfully.")

# Load label encoder
label_encoder = joblib.load('models/labels.pkl')
print("Label encoder loaded successfully.")

# Function to preprocess image for prediction
def preprocess_image(image_path, img_size=128):
    img = cv2.imread(image_path)
    img = cv2.resize(img, (img_size, img_size))  # Resize image
    img = img / 255.0  # Normalize
    img = np.expand_dims(img, axis=0)  # Add batch dimension
    return img

# Function to predict disease from an image
def predict_disease(image_path):
    img = preprocess_image(image_path)
    
    # Set tensor to the input details of the model
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # Set input tensor
    interpreter.set_tensor(input_details[0]['index'], img.astype(np.float32))
    
    # Invoke the interpreter
    interpreter.invoke()
    
    # Get the output prediction
    features = interpreter.get_tensor(output_details[0]['index'])
    
    # Predict disease using Random Forest
    disease_prediction = rf_classifier.predict(features)
    
    # Map numeric prediction to disease label
    disease_label = label_encoder.inverse_transform(disease_prediction)
    
    return disease_label[0]

# Route for the home page
@app.route('/')
def home():
    return render_template('index.html')

# Route to handle image upload and prediction
@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'})
    
    # Save the uploaded image
    image_path = os.path.join('static', file.filename)
    file.save(image_path)
    
    # Predict disease
    predicted_disease = predict_disease(image_path)
    
    return jsonify({'predicted_disease': predicted_disease})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))
