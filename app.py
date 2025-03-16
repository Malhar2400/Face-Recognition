import streamlit as st
import cloudinary
import cloudinary.uploader
import face_recognition
import numpy as np
import cv2
from pymongo import MongoClient

# ✅ Cloudinary Configuration
cloudinary.config(
    cloud_name=st.secrets["CLOUDINARY_CLOUD_NAME"],
    api_key=st.secrets["CLOUDINARY_API_KEY"],
    api_secret=st.secrets["CLOUDINARY_API_SECRET"]
)

# ✅ MongoDB Connection
client = MongoClient(st.secrets["MONGODB_URI"])
db = client["face_recognition_db"]
collection = db["faces"]

# 📌 Extract face encoding from an image
def extract_faces(image):
    encodings = face_recognition.face_encodings(image)
    return encodings[0] if encodings else None

# 📌 Find matching faces in the database
def find_matching_faces(uploaded_encoding):
    stored_faces = collection.find({})
    matched_images = []

    for face_data in stored_faces:
        stored_encoding = np.array(face_data["face_encoding"])
        
        # Compute face distance (lower is better)
        distance = face_recognition.face_distance([stored_encoding], uploaded_encoding)[0]
        
        confidence = (1 - distance) * 100 

        if confidence >= 45:  # Only include matches with confidence >= 45%
            matched_images.append((face_data["image_url"], confidence))

    # Sort by confidence (highest first)
    return sorted(matched_faces, key=lambda x: x[1], reverse=True)

# 📌 Streamlit UI
def main():
    st.title("Face Recognition - Hackathon")
    st.write("Upload a photo to find matching faces in the database.")

    # File uploader
    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

    if uploaded_file:
        # Convert uploaded image to OpenCV format
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        
        # Convert to RGB for display
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Display uploaded image
        st.image(rgb_image, caption="Uploaded Image", use_container_width=True)
        
        # Extract face encoding
        uploaded_encoding = extract_faces(rgb_image)
        
        if uploaded_encoding is None:
            st.error("No face detected. Please upload a clear image with a visible face.")
        else:
            # Match against database
            matched_faces = find_matching_faces(uploaded_encoding)
            
            if matched_faces:
                st.success(f"✅ {len(matched_faces)} High-Confidence Match(es) Found!")

                # Display matched images in a grid
                cols = st.columns(3) 
                for i, (img_url, confidence) in enumerate(matched_faces):
                    with cols[i % 3]: 
                        st.image(img_url, caption=f"Confidence: {confidence:.2f}%", use_container_width=True)
            else:
                st.warning("❌ No high-confidence matches found (above 45%).")

# Run the app
if __name__ == "__main__":
    main()
