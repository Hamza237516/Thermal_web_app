from fastapi import FastAPI, UploadFile, File
from PIL import Image
import torch
from transformers import DetrImageProcessor, DetrForObjectDetection
import io
import config  # Import your settings

app = FastAPI(title=config.API_TITLE)

# --- 1. Health Check Endpoint ---
# This is what turns your dashboard status light GREEN
@app.get("/health")
async def health_check():
    return {
        "status": "online",
        "model_loaded": config.MODEL_PATH,
        "device": str(config.DEVICE)
    }

# --- 2. Model Initialization ---
print(f"⏳ Loading {config.MODEL_PATH}...")

# Load the base processor
processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50")

# Load the architecture
model = DetrForObjectDetection.from_pretrained(
    "facebook/detr-resnet-50",
    num_labels=91, 
    ignore_mismatched_sizes=True
)

# Load your custom thermal weights
# Note: This is where the file 'thermal_detr_epoch_15.pth' is required!
try:
    model.load_state_dict(torch.load(config.MODEL_PATH, map_location=config.DEVICE))
    model.to(config.DEVICE)
    model.eval()
    print("✅ Custom Thermal Model Ready!")
except FileNotFoundError:
    print(f"❌ ERROR: {config.MODEL_PATH} not found. Please add it to your folder.")

# --- 3. Prediction Logic ---
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    # Read and process the image
    image_bytes = await file.read()
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    inputs = processor(images=image, return_tensors="pt").to(config.DEVICE)
    
    with torch.no_grad():
        outputs = model(**inputs)
    
    # Convert outputs to detections
    target_sizes = torch.tensor([image.size[::-1]])
    results = processor.post_process_object_detection(
        outputs, 
        target_sizes=target_sizes, 
        threshold=config.CONFIDENCE_THRESHOLD
    )[0]
    
    detections = []
    for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
        detections.append({
            "label": model.config.id2label[label.item()],
            "confidence": round(score.item(), 3),
            "box": [round(i, 2) for i in box.tolist()]
        })
    
    return {"detections": detections}