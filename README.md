# 🔥 Thermal Tracking DETR Dashboard

A real-time object detection dashboard built for analyzing long-wave infrared (LWIR) thermal imagery. 

This repository contains the deployment codebase (Frontend and API). The model architecture and training loops (fine-tuned on the Teledyne FLIR ADAS dataset) can be found in the [Training Repository](https://github.com/Hamza237516/Thermal-Tracking-DETR).

## Tech Stack
* **Backend:** FastAPI, PyTorch, HuggingFace Transformers
* **Frontend:** Streamlit
* **Model:** DETR-ResNet50 (15 Epochs, Thermal Fine-Tuned)

## How to Run Locally
1. Clone this repository.
2. Download the custom model weights (`.pth`) and place them in the root directory.
3. Install dependencies: `pip install -r requirements.txt`
4. Start the backend: `uvicorn api:app --reload`
5. Start the frontend: `streamlit run app.py`