🔥 Thermal_web_app
### *Real-Time Autonomous Object Detection for Infrared Systems*

[![Python 3.13](https://img.shields.io/badge/Python-3.13-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/Backend-FastAPI-009688?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![Streamlit](https://img.shields.io/badge/Frontend-Streamlit-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io/)
[![PyTorch](https://img.shields.io/badge/AI-PyTorch-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)

## 📌 Project Overview
This repository, **Thermal_web_app**, serves as the **Deployment & Dashboard** layer for an advanced object detection system tailored for long-wave infrared (LWIR) environments. It houses the core application logic (`api.py` and `app.py`) necessary to visualize and interact with the DETR model.

The system utilizes a **DEtection TRansformer (DETR)** architecture fine-tuned on the **Teledyne FLIR ADAS Dataset** to localize vehicles and pedestrians in challenging conditions like total darkness or smoke.

> **Note:** This repository is specifically for deployment. The core training logic and model weight generation are maintained in the [Training Repository](https://github.com/Hamza237516/Thermal-Tracking-DETR).

## 🏗️ Decoupled Architecture
To ensure high scalability, the system is split into two primary components:
*   **Neural Backend (`api.py`):** A **FastAPI** server that hosts the **DETR-ResNet50** model and handles image preprocessing.
*   **Telemetry Dashboard (`app.py`):** A **Streamlit** user interface for real-time visualization, confidence filtering, and performance metrics.

## 🚀 Key Features
*   **Thermal Signal Logic:** Optimized for 1-channel thermal heat maps.
*   **Dynamic Thresholding:** Adjust detection sensitivity via the UI sidebar.
*   **Performance Telemetry:** Real-time monitoring of inference latency and object identification.

## 📊 Performance Benchmarks
*   **Inference Latency:** ~0.45s (Benchmarked on local MacBook hardware).
*   **Concurrency:** Robust localization for **35+ concurrent objects** per frame.
*   **Model Training:** Fine-tuned for **15 Epochs** for specialized thermal reasoning.

## 🛠️ Tech Stack
*   **Core AI:** PyTorch, HuggingFace Transformers, Timm.
*   **Deployment:** FastAPI, Uvicorn, Python 3.13.
*   **Frontend:** Streamlit, PIL (Pillow), Requests.

## ⚙️ Setup & Installation
1.  **Clone the Repository:**
    ```bash
    git clone [https://github.com/Hamza237516/Thermal_web_app.git](https://github.com/Hamza237516/Thermal_web_app.git)
    cd Thermal_web_app
    ```
2.  **Initialize Virtual Environment:**
    ```bash
    python3 -m venv venv
    source venv/bin/activate
    pip install -r requirements.txt
    ```
3.  **Local Execution:**
    *   **Start Backend:** `uvicorn api:app --reload`
    *   **Start Frontend:** `streamlit run app.py`

---
**Developed by [Hamza Mehmood](https://github.com/Hamza237516)**
*NIT Alumnus | Aspiring Graduate Student in AI*
