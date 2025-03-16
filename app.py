import streamlit as st
import cloudinary
import cloudinary.uploader
import numpy as np
import cv2
from pymongo import MongoClient
from deepface import DeepFace
import tempfile
import os

# ✅ Cloudinary Configuration
cloudinary.config(
    cloud_name="djtgzknrm",
    api_key="489942253832734",
    api_secret="pE_6C8nMawyU9dhthyIy0B3C9yU"
)

# ✅ MongoDB Connection
client = MongoClient("mongodb+srv://yash:yash2424@cluster.cyo01.mongodb.net/forumDB?retryWrites=true&w=majority")
db = client["face_recognition_db"]
collection = db["faces"]

def extract_face_embedding(image_path):
    try:
        # DeepFace uses VGG-Face by default
        embedding_obj = DeepFace.represent(
            img_path=image_path,
            model_name="VGG-Face", 
            detector_backend="retinaface"
        )
        # Return the embedding vector
        return embedding_obj[0]["embedding"]
    except Exception as e:
        st.error(f"Error extracting face: {str(e)}")
        return None

def find_matching_faces(uploaded_embedding):
    stored_faces = collection.find({})
    matched_images = []
    
    for face_data in stored_faces:
        stored_embedding = np.array(face_data["face_embedding"])
        
        # Compute cosine similarity
        similarity = np.dot(uploaded_embedding, stored_embedding) / (
            np.linalg.norm(uploaded_embedding) * np.linalg.norm(stored_embedding)
        )
        
        # Convert similarity to confidence percentage
        confidence = similarity * 100
        
        if confidence >= 45:
            matched_images.append((face_data["image_url"], confidence))
    
    return sorted(matched_images, key=lambda x: x[1], reverse=True)  # Sort by confidence (highest first)

# ✅ Streamlit UI
st.title("Face Recognition - Hackathon")
st.write("Upload a photo to find matching faces in the database.")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    # Save uploaded file temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp_file:
        temp_filename = temp_file.name
        temp_file.write(uploaded_file.getvalue())
    
    # Display the uploaded image
    image = cv2.imread(temp_filename)
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    st.image(rgb_image, caption="Uploaded Image", use_container_width=True)
    
    # Extract embedding
    uploaded_embedding = extract_face_embedding(temp_filename)
    
    # Clean up temporary file
    os.unlink(temp_filename)
    
    if uploaded_embedding is None:
        st.error("No face detected. Please upload a clear image with a visible face.")
    else:
        # Match against database
        matched_faces = find_matching_faces(uploaded_embedding)
        
        if matched_faces:
            st.success(f"✅ {len(matched_faces)} High-Confidence Match(es) Found!")
            
            cols = st.columns(3)
            
            for i, (img_url, confidence) in enumerate(matched_faces):
                with cols[i % 3]:
                    st.image(img_url, caption=f"Confidence: {confidence:.2f}%", use_container_width=True)
        else:
            st.warning("❌ No high-confidence matches found (above 45%).")