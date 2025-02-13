import os
import numpy as np
import cv2
import joblib
import tensorflow as tf
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from starlette.middleware.cors import CORSMiddleware
import shutil

# Initialize FastAPI app
app = FastAPI()

# Enable CORS for React frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "https://your-frontend-url.com"],  
    allow_credentials=True,
    allow_methods=["POST", "OPTIONS"],  # Ensure OPTIONS is allowed
    allow_headers=["*"],
)


# Paths to model and encoder
cnn_model_path = 'models/model_quantized.tflite'
rfc_model_path = 'models/rfc2.pkl'
label_encoder_path = 'models/labels.pkl'

# Load the models and encoder
interpreter = tf.lite.Interpreter(model_path=cnn_model_path)
interpreter.allocate_tensors()
print("CNN model loaded successfully.")

rf_classifier = joblib.load(rfc_model_path)
print("Random Forest model loaded successfully.")

label_encoder = joblib.load(label_encoder_path)
print("Label encoder loaded successfully.")

# Function to preprocess image
def preprocess_image(image_path, img_size=128):
    img = cv2.imread(image_path)
    img = cv2.resize(img, (img_size, img_size))  # Resize image
    img = img / 255.0  # Normalize
    img = np.expand_dims(img, axis=0)  # Add batch dimension
    return img

# Function to predict disease
def predict_disease(image_path):
    img = preprocess_image(image_path)
    
    # Set input tensor
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    interpreter.set_tensor(input_details[0]['index'], img.astype(np.float32))
    
    # Invoke the model
    interpreter.invoke()
    
    # Extract features
    features = interpreter.get_tensor(output_details[0]['index'])
    
    # Predict using Random Forest
    disease_prediction = rf_classifier.predict(features)
    
    # Decode the predicted label
    disease_label = label_encoder.inverse_transform(disease_prediction)
    
    return disease_label[0]

# Home route
@app.get("/")
def home():
    return {"message": "FastAPI backend is running"}

# Image upload and prediction route
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    file_location = f"static/{file.filename}"
    
    # Save uploaded file
    with open(file_location, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    # Get prediction
    predicted_disease = predict_disease(file_location)
    
    # Return JSON response
    return JSONResponse(content={"predicted_disease": predicted_disease})
