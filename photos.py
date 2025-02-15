import os
import cloudinary
import cloudinary.uploader
import face_recognition
import numpy as np
from pymongo import MongoClient

# ✅ Cloudinary Configuration
cloudinary.config(
    cloud_name="djtgzknrm",
    api_key="489942253832734",
    api_secret="pE_6C8nMawyU9dhthyIy0B3C9yU"
)

# ✅ MongoDB Connection
client = MongoClient("mongodb://localhost:27017/")
db = client["face_recognition_db"]
collection = db["faces"]

# 📌 Extract face encoding from an image
def extract_faces(image_path):
    image = face_recognition.load_image_file(image_path)
    encodings = face_recognition.face_encodings(image)
    print(f"ℹ️ Found {len(encodings)} face(s) in {image_path}")
    return encodings[0] if encodings else None  # Return the first face found

# 📌 Upload image to Cloudinary & store encoding in MongoDB
def upload_and_store(image_path):
    encoding = extract_faces(image_path)
    
    if encoding is None:
        print(f"❌ No face detected in {image_path}")
        return
    
    # Upload image to Cloudinary
    response = cloudinary.uploader.upload(image_path, folder="hackathon_faces")
    cloudinary_url = response["secure_url"]

    # Store face encoding & Cloudinary URL in MongoDB
    encoding_list = encoding.tolist()
    collection.insert_one({"face_encoding": encoding_list, "image_url": cloudinary_url})
    
    print(f"✅ Uploaded {image_path} and stored face encoding.")

# 📌 Process all images in the "Photos" directory
def process_folder(folder_path):
    for filename in os.listdir(folder_path):
        image_path = os.path.join(folder_path, filename)
        upload_and_store(image_path)
        

# 📌 Run the script
if __name__ == "__main__":
    process_folder("Photos")  # Replace with your images folder
