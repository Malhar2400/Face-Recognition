import streamlit as st
import cloudinary
import cloudinary.uploader
import tempfile
import cv2
import numpy as np
import json

# Load Cloudinary credentials from Streamlit secrets
cloudinary.config(
    cloud_name=st.secrets["cloudinary"]["cloud_name"],
    api_key=st.secrets["cloudinary"]["api_key"],
    api_secret=st.secrets["cloudinary"]["api_secret"]
)

# Streamlit App UI
st.title("Hackathon Selfie Registration")
st.write("Click a selfie to register and upload to Cloudinary.")

# Capture image
img_file_buffer = st.camera_input("Take a picture")

if img_file_buffer is not None:
    # Convert to OpenCV format
    img_bytes = img_file_buffer.getvalue()
    np_arr = np.frombuffer(img_bytes, np.uint8)
    img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    # Save temporarily
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".jpg")
    temp_file_path = temp_file.name
    cv2.imwrite(temp_file_path, img)

    st.image(img, caption="Captured Image", use_column_width=True)

    # Upload to Cloudinary
    response = cloudinary.uploader.upload(temp_file_path)
    cloudinary_url = response["secure_url"]

    st.success("Image uploaded successfully!")
    st.write(f"[View Image]({cloudinary_url})")
