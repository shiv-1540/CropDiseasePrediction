from fastapi import FastAPI, UploadFile, File
import google.generativeai as genai
from PIL import Image
from io import BytesIO
import base64
from fastapi.middleware.cors import CORSMiddleware


# FastAPI instance
app = FastAPI()
# Allow all origins (use specific origins in production)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Change to specific origins for security
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configure API key
genai.configure(api_key="AIzaSyAaj7V1UIbiLpOPQ2_A7oK6sKFHSdt81Lg")

# Define Model Configuration
generation_config = {
    "temperature": 0.4,
    "top_p": 1,
    "top_k": 32,
    "max_output_tokens": 4096,
}
safety_settings = [
    {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
    {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
    {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
    {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
]
model = genai.GenerativeModel(
    model_name="gemini-1.5-flash",
    generation_config=generation_config,
    safety_settings=safety_settings,
)

def process_image(image_bytes):
    image_base64 = base64.b64encode(image_bytes).decode("utf-8")

    image_parts = [
        {"mime_type": "image/png", "data": image_base64}
    ]
    prompt_parts = [
        "You are a professional Plant disease detector. I'll provide an image of a leaf of a plant. Identify any disease in the plant and provide a structured response in the following format:\n\n",
        "Predicted Plant Disease: [Include name of disease, details about symptoms, affected parts, etc.]\n",
        "Precautions: [List 2-3 precautionary measures]\n",
        "Remedies: [Provide 2-3 treatment methods]\n",
        image_parts[0]
    ]

    response = model.generate_content(prompt_parts)
    return response.text

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    image_bytes = await file.read()
    result = process_image(image_bytes)
    return {"prediction": result}

@app.get("/")
async def root():
    return {"message": "Plant Disease Detection API is running"}
