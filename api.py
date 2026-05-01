from fastapi import FastAPI, UploadFile, File
from PIL import Image
import torch
from transformers import DetrImageProcessor, DetrForObjectDetection
import io

app = FastAPI(title="Thermal Tracking API")

# Setup the device for your MacBook
device = torch.device("cpu") 

print("⏳ Loading Custom Thermal DETR Model...")
processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50")
model = DetrForObjectDetection.from_pretrained(
    "facebook/detr-resnet-50",
    num_labels=91, 
    ignore_mismatched_sizes=True
)

# Load your custom weights from Kaggle
model.load_state_dict(torch.load("thermal_detr_epoch_15.pth", map_location=device))
model.to(device)
model.eval()

print("✅ Custom Thermal Model Ready!")

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    image_bytes = await file.read()
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    inputs = processor(images=image, return_tensors="pt").to(device)
    
    with torch.no_grad():
        outputs = model(**inputs)
    
    target_sizes = torch.tensor([image.size[::-1]])
    results = processor.post_process_object_detection(outputs, target_sizes=target_sizes, threshold=0.5)[0]
    
    detections = []
    for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
        detections.append({
            "label": model.config.id2label[label.item()],
            "confidence": round(score.item(), 3),
            "box": [round(i, 2) for i in box.tolist()]
        })
    return {"detections": detections}