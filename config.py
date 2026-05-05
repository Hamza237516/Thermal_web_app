import torch

# --- Model Settings ---
MODEL_PATH = "thermal_detr_epoch_15.pth"
DEVICE = torch.device("cpu") 

# --- Confidence Settings (Unified) ---
# Your Backend (api.py) looks for this:
CONFIDENCE_THRESHOLD = 0.50 
# Your Frontend (app.py) looks for this:
DEFAULT_CONFIDENCE = 0.50

# --- Server Settings ---
HOST = "127.0.0.1"
PORT = 8000
API_TITLE = "Thermal Tracking API"