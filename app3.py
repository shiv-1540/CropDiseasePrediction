import torch
import numpy as np
from fastapi import FastAPI, File, UploadFile
from PIL import Image
import CNN
import torch.nn.functional as F  # For softmax
import gdown


# Initialize FastAPI app
app = FastAPI()

# Google Drive file ID (Replace with your actual file ID)
file_id = "17eu7C-dnftoeYxTuZ-3zdIx-CXkbLlBs"
url = f"https://drive.google.com/uc?id={file_id}"

# Download the model
gdown.download(url, "plant_disease_model_1_latest.pt", quiet=False)
# Load the trained model
model = CNN.CNN(39)
model.load_state_dict(torch.load("plant_disease_model_1_latest.pt", map_location=torch.device("cpu")))
model.eval()

# Prediction function with confidence
def predict_disease(image: Image.Image):
    # Resize and normalize image
    image = image.resize((224, 224))
    image_np = np.array(image).astype(np.float32) / 255.0
    input_tensor = torch.tensor(image_np).permute(2, 0, 1).unsqueeze(0)  # (C, H, W) format with batch dimension

    # Perform inference
    with torch.no_grad():
        output = model(input_tensor)
        probabilities = F.softmax(output, dim=1)  # Convert logits to probabilities

    # Get predicted class and confidence score
    pred_index = torch.argmax(probabilities).item()
    confidence = probabilities[0, pred_index].item()  # Confidence of predicted class

    return pred_index, confidence

# API endpoint for prediction
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    image = Image.open(file.file)
    pred_class, confidence = predict_disease(image)
    return {"prediction": pred_class, "confidence": confidence}

# Run the app
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
