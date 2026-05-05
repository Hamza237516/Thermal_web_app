import streamlit as st
import requests
from PIL import Image, ImageDraw
import time
import config  # Using our central configuration

# --- Page Configuration ---
st.set_page_config(
    page_title="Thermal DETR Monitor",
    page_icon="🔥",
    layout="wide"
)

# --- Custom Dashboard Styling ---
st.markdown("""
    <style>
    .main { background-color: #0e1117; }
    .stMetric { background-color: #1e2130; padding: 15px; border-radius: 10px; border: 1px solid #3e4259; }
    div[data-testid="stExpander"] { border: none !important; box-shadow: none !important; }
    </style>
    """, unsafe_allow_html=True)

# --- Backend Health Check ---
def check_backend_health():
    """Pings the /health endpoint we added to api.py"""
    try:
        url = f"http://{config.HOST}:{config.PORT}/health"
        response = requests.get(url, timeout=1)
        return response.status_code == 200
    except:
        return False

# --- Sidebar Configuration ---
with st.sidebar:
    st.title("⚙️ Parameters")
    st.markdown("Adjust the AI's sensitivity in real-time.")
    
    # Using the default confidence from our config
    conf_threshold = st.slider("Confidence Threshold", 0.0, 1.0, config.DEFAULT_CONFIDENCE, 0.05)
    
    st.divider()
    st.subheader("📊 System Status")
    
    # DYNAMIC STATUS CHECK
    is_online = check_backend_health()
    
    if is_online:
        st.success("Backend: Online")
        st.info(f"Model: {config.MODEL_PATH}")
        st.caption("Fine-tuned on FLIR Thermal Dataset (15 Epochs)")
    else:
        st.error("Backend: Offline")
        st.warning("⚠️ Action Required: Start the FastAPI server in your terminal.")
        st.stop() # Prevents the UI from loading further if the backend is down

# --- Header Section ---
st.title("🔥 Thermal-Tracking-DETR Dashboard")
st.markdown("### Intelligent Heat Signature Recognition")

# --- Main Layout ---
col_in, col_out = st.columns([1, 1], gap="large")

with col_in:
    st.subheader("📥 Input Stream")
    uploaded_file = st.file_uploader("Drop thermal imagery here...", type=["jpg", "jpeg", "png"])
    
    if uploaded_file:
        input_img = Image.open(uploaded_file).convert("RGB")
        st.image(input_img, caption="Original Signal", use_container_width=True)

with col_out:
    st.subheader("🎯 Neural Analysis")
    
    if uploaded_file:
        if st.button("Analyze Thermal Signature", type="primary"):
            with st.spinner("Crunching tensors..."):
                try:
                    start = time.time()
                    
                    # API Call using config host/port
                    payload = {"file": uploaded_file.getvalue()}
                    predict_url = f"http://{config.HOST}:{config.PORT}/predict"
                    response = requests.post(predict_url, files=payload)
                    data = response.json()["detections"]
                    
                    elapsed = round(time.time() - start, 3)
                    
                    # Filtering and Overlay
                    processed_img = input_img.copy()
                    draw = ImageDraw.Draw(processed_img)
                    valid_dets = [d for d in data if d['confidence'] >= conf_threshold]
                    
                    for det in valid_dets:
                        box = det["box"]
                        # Draw aesthetic bounding boxes
                        draw.rectangle(box, outline="#FF4B4B", width=4)
                        # Label background
                        draw.rectangle([box[0], box[1]-20, box[0]+80, box[1]], fill="#FF4B4B")
                        draw.text((box[0]+5, box[1]-18), f"{det['label']} {det['confidence']}", fill="white")

                    st.image(processed_img, caption=f"Detection Overlay (Latency: {elapsed}s)", use_container_width=True)
                    
                    # Performance Metrics
                    m1, m2 = st.columns(2)
                    m1.metric("Objects Identified", len(valid_dets))
                    m2.metric("Inference Latency", f"{elapsed}s")
                    
                except Exception as e:
                    st.error(f"Prediction Failed: The model could not process the image. Error: {e}")
    else:
        st.info("Awaiting input file for telemetry analysis.")

st.divider()
st.caption("Autonomous Thermal Recognition System | Local Inference Engine")