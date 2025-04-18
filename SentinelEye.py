import streamlit as st
import cv2
import tempfile
import os
from datetime import datetime
from transformers import BlipProcessor, BlipForConditionalGeneration
import torch
from PIL import Image
import yagmail
import pandas as pd
import asyncio
import contextlib

# App config
st.set_page_config(page_title="SentinelEye | Threat Detection", layout="wide")

# Constants
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
THREAT_KEYWORDS = ["weapon", "gun", "knife", "pistol", "rifle", "firearm"]
THREAT_DIR = "threat_captures"
os.makedirs(THREAT_DIR, exist_ok=True)

# Email setup (Optional - make sure network is reachable)
EMAIL_SENDER = "hackathon0samai@gmail.com"              # Replace with sender email
EMAIL_PASSWORD = "data@run"                             # Replace with app password
EMAIL_RECEIVER = "sarmimala.saikia@gmail.com"           # Replace with your email

# --- AsyncIO fix for Streamlit + Python 3.11 ---
with contextlib.suppress(RuntimeError):
    asyncio.get_running_loop()

# Load model
@st.cache_resource
def load_model():
    processor = BlipProcessor.from_pretrained(
        "Salesforce/blip-image-captioning-base", use_fast=True
    )
    model = BlipForConditionalGeneration.from_pretrained(
        "Salesforce/blip-image-captioning-base"
    ).to(DEVICE)
    return processor, model

processor, model = load_model()

# Style
st.markdown("""
    <style>
    .threat-banner {
        position: fixed;
        top: 60px;
        width: 100%;
        background-color: #ff4b4b;
        color: white;
        font-size: 30px;
        font-weight: bold;
        text-align: center;
        padding: 10px;
        z-index: 9999;
        border-radius: 0 0 10px 10px;
    }
    </style>
""", unsafe_allow_html=True)

# Sidebar mode selector
mode = st.sidebar.radio("Choose Input Mode", ["📷 Webcam", "📹 Upload Video"])

# Threat display
st.title("🔐 SentinelEye – Real-Time Threat Detection")
alert_banner = st.empty()
frame_display = st.empty()

# Threat logs
detected_threats = []

def log_threat_to_file(timestamp, caption):
    with open(os.path.join(THREAT_DIR, "threat_log.txt"), "a") as log_file:
        log_file.write(f"{timestamp} - {caption}\n")

def detect_threat(frame):
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as temp:
        cv2.imwrite(temp.name, frame)
        image = Image.open(temp.name).convert("RGB")

    inputs = processor(image, return_tensors="pt").to(DEVICE)
    out = model.generate(**inputs)
    caption = processor.decode(out[0], skip_special_tokens=True)

    if any(word in caption.lower() for word in THREAT_KEYWORDS):
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        alert_banner.markdown(f"<div class='threat-banner'>⚠️ THREAT DETECTED: {caption.upper()}</div>", unsafe_allow_html=True)
        filename = os.path.join(THREAT_DIR, f"threat_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg")
        cv2.imwrite(filename, frame)
        detected_threats.append({"Time": timestamp, "Description": caption})
        log_threat_to_file(timestamp, caption)

        try:
            yag = yagmail.SMTP(user=EMAIL_SENDER, password=EMAIL_PASSWORD)
            yag.send(to=EMAIL_RECEIVER, subject="⚠️ Threat Detected", contents=[caption, filename])
        except:
            pass
    else:
        alert_banner.empty()

    return caption

# Webcam mode
if mode == "📷 Webcam":
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        st.error("Cannot access webcam.")
    else:
        st.info("Press `Stop` to end webcam feed.")
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            detect_threat(frame)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_display.image(frame_rgb, channels="RGB", use_container_width=False, width=640)
            if not st.runtime.scriptrunner.running_with_streamlit:
                break
        cap.release()

# Video upload mode
elif mode == "📹 Upload Video":
    uploaded_video = st.file_uploader("Upload a video file", type=["mp4", "avi", "mov"])
    if uploaded_video is not None:
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_video.read())
        cap = cv2.VideoCapture(tfile.name)

        while cap.isOpened():
            ret, frame = cap.read()
            max_width = 640
            h, w = frame.shape[:2]
            if w > max_width:
                scale = max_width / w
                frame = cv2.resize(frame, (int(w * scale), int(h * scale)))

            if not ret:
                break
            detect_threat(frame)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_display.image(frame_rgb, channels="RGB", use_container_width=False, width=640)
        cap.release()

# 📊 Threat Log Table
if detected_threats:
    st.subheader("📋 Threat Log")
    df = pd.DataFrame(detected_threats)
    st.dataframe(df, use_container_width=True)

# 📁 Log download button
log_path = os.path.join(THREAT_DIR, "threat_log.txt")
if os.path.exists(log_path):
    with open(log_path, "r") as file:
        st.sidebar.download_button("⬇️ Download Threat Log", file, file_name="threat_log.txt")
