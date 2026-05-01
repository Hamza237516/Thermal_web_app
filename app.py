import streamlit as st
import requests
from PIL import Image, ImageDraw, ImageFont
import time

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

# --- Sidebar Configuration ---
with st.sidebar:
    st.title("⚙️ Parameters")
    st.markdown("Adjust the AI's sensitivity in real-time.")
    
    # Confidence Slider
    conf_threshold = st.slider("Confidence Threshold", 0.0, 1.0, 0.5, 0.05)
    
    st.divider()
    st.subheader("📊 System Status")
    st.success("Backend: Online")
    st.info("Model: DETR-ResNet50")
    st.caption("Fine-tuned on FLIR Thermal Dataset (15 Epochs)")

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
                    # Tracking inference speed
                    start = time.time()
                    
                    # API Call
                    payload = {"file": uploaded_file.getvalue()}
                    response = requests.post("http://127.0.0.1:8000/predict", files=payload)
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
                    st.error(f"Connection Lost: Ensure FastAPI is running on port 8000. Error: {e}")
    else:
        st.info("Awaiting input file for telemetry analysis.")

st.divider()
st.caption("Autonomous Thermal Recognition System | Local Inference Engine")