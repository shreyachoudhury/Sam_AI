pip install yagmail
pip install streamlit
import os
from datetime import datetime
import yagmail

import streamlit as st
import cv2
import tempfile
import torch
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration
import numpy as np
import asyncio
import contextlib

# ---------------------- SETUP EMAIL CREDENTIALS -----------------------
EMAIL_SENDER = "hackathon0samai@gmail.com"           # Replace with sender email
EMAIL_PASSWORD = "data@run"            # Replace with app password
EMAIL_RECEIVER = "sarmimala.saikia@gmail.com"              # Replace with your email

# ---------------------- FOLDER FOR THREAT LOGS ------------------------
LOG_DIR = "threat_logs"
os.makedirs(LOG_DIR, exist_ok=True)


# --- AsyncIO fix for Streamlit + Python 3.11 ---
with contextlib.suppress(RuntimeError):
    asyncio.get_running_loop()


# --- Load BLIP model + processor ---
@st.cache_resource
def load_model():
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base", use_fast=True)
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
    model.to("cuda" if torch.cuda.is_available() else "cpu")
    return processor, model

processor, model = load_model()

# --- Caption prediction ---
def generate_caption(image):
    inputs = processor(images=image, return_tensors="pt").to(model.device)
    out = model.generate(**inputs)
    caption = processor.decode(out[0], skip_special_tokens=True)
    return caption

# ---------------------- LOG THREAT EVENT ------------------------------
def log_threat(frame, caption):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"threat_{timestamp}.jpg"
    filepath = os.path.join(LOG_DIR, filename)
    cv2.imwrite(filepath, frame)

    with open(os.path.join(LOG_DIR, "log.txt"), "a") as log_file:
        log_file.write(f"{timestamp} - {caption}\n")

    return filepath, timestamp

# ---------------------- SEND EMAIL ALERT ------------------------------
def send_email_alert(caption, image_path):
    try:
        yag = yagmail.SMTP(user=EMAIL_SENDER, password=EMAIL_PASSWORD)
        subject = "⚠️ Weapon Threat Detected"
        contents = [
            f"A threat was detected:\n\n{caption}",
            image_path
        ]
        yag.send(to=EMAIL_RECEIVER, subject=subject, contents=contents)
    except Exception as e:
        st.error(f"Email sending failed: {e}")

# ---  Stramlit UI ---
st.title("🔍 Weapon Detection with Vision-Language AI")
st.markdown("Detects threats (guns, knives, etc.) in real-time and sends alerts.")

mode = st.radio("Choose input method:", ["📷 Webcam", "📁 Upload Video"])

# --- Video Capture Logic ---
def process_video_stream(source):
    cap = cv2.VideoCapture(source)
    stframe = st.empty()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Convert to RGB PIL image
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(frame_rgb)

        # Generate caption
        caption = generate_caption(pil_image)

        # Detect threat
        threat_keywords = ["gun", "weapon", "knife", "rifle"]
        is_threat = any(word in caption.lower() for word in threat_keywords)

        # Create threat label
        label = f"⚠️ THREAT DETECTED: {caption.upper()}" if is_threat else f"✅ Safe: {caption.capitalize()}"

        # Overlay label at the top
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 1.2 if is_threat else 0.9
        color = (0, 0, 255) if is_threat else (0, 255, 0)
        thickness = 3 if is_threat else 2
        (text_width, text_height), _ = cv2.getTextSize(label, font, font_scale, thickness)

        cv2.rectangle(frame, (0, 0), (text_width + 30, text_height + 20), (0, 0, 0), -1)
        cv2.putText(frame, label, (10, text_height + 10), font, font_scale, color, thickness, cv2.LINE_AA)

        # Show frame
        stframe.image(frame, channels="BGR")

        # Log and alert if threat
        if is_threat:
            filepath, timestamp = log_threat(frame, caption)
            send_email_alert(caption, filepath)

    cap.release()

# --- Webcam Mode ---
if mode == "📷 Webcam":
    st.info("Make sure your webcam is available and not used by another app.")
    process_video_stream(0)

# --- Video Upload Mode ---
else:
    uploaded_file = st.file_uploader("Upload a video file", type=["mp4", "avi", "mov"])
    if uploaded_file is not None:
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_file.read())
        process_video_stream(tfile.name)
