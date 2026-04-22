# train.py
import os
import json
import numpy as np
import cv2
import time
from datetime import datetime
from insightface.app import FaceAnalysis
from typing import List
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
import uvicorn
from pymongo import MongoClient
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# MongoDB Setup
MONGO_URI = os.getenv("MONGO_URI")
client = MongoClient(MONGO_URI)
db_mongo = client["pg_axis"]
collection_trained = db_mongo["face_trained_data"]

TRAIN_DIR = "train_images"
TRAINED_DIR = "Trained"

# Quality thresholds
TRAINING_QUALITY_THRESHOLD = 65.0 # Minimum unified score to allow training

# FastAPI instance
app = FastAPI(title="Face Trainer Service")

# Initialize InsightFace
print("Loading face analysis model...")
face_app = FaceAnalysis(name="buffalo_l", providers=['CPUExecutionProvider'])
face_app.prepare(ctx_id=-1)

def get_image_quality(img, face):
    """Calculate detection confidence and sharpness on the face region."""
    det_score = face.det_score
    
    # Crop to the face bounding box
    bbox = face.bbox.astype(int)
    x1, y1, x2, y2 = max(0, bbox[0]), max(0, bbox[1]), min(img.shape[1], bbox[2]), min(img.shape[0], bbox[3])
    face_img = img[y1:y2, x1:x2]
    
    if face_img.size == 0:
        return float(det_score), 0.0

    gray_face = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
    blur_score = cv2.Laplacian(gray_face, cv2.CV_64F).var()
    brightness = np.mean(gray_face)
    contrast = np.std(gray_face)
    
    # --- Unified Quality Score Calculation (0-100) ---
    # 1. Detection (Confidence) - Weight 40%
    n_det = det_score * 100
    
    # 2. Sharpness (Blur) - Weight 30% (Perfect sharpness starts at ~300)
    n_blur = min(100, (blur_score / 300.0) * 100)
    
    # 3. Brightness - Weight 15% (Ideal brightness ~110)
    n_bright = max(0, 100 - abs(brightness - 110) * 0.8)
    
    # 4. Contrast - Weight 15% (Ideal contrast > 60)
    n_contrast = min(100, (contrast / 60.0) * 100)
    
    unified_score = (n_det * 0.40) + (n_blur * 0.30) + (n_bright * 0.15) + (n_contrast * 0.15)
    
    return float(round(unified_score, 2)), {
        "det": float(round(det_score, 3)),
        "blur": float(round(blur_score, 2)),
        "brightness": float(round(brightness, 2)),
        "contrast": float(round(contrast, 2))
    }

def save_trained_image(name, image):
    """Save training dataset images into Trained/<name>/"""
    person_dir = os.path.join(TRAINED_DIR, name)
    os.makedirs(person_dir, exist_ok=True)
    count = len(os.listdir(person_dir)) + 1
    path = os.path.join(person_dir, f"{count}.jpg")
    cv2.imwrite(path, image)
    return path

def extract_embeddings(img):
    """Detect faces and return embeddings."""
    faces = face_app.get(img)
    if len(faces) == 0:
        return []
    
    # Pick the largest face (main subject)
    largest_face = max(faces, key=lambda f: (f.bbox[2]-f.bbox[0]) * (f.bbox[3]-f.bbox[1]))
    
    emb = largest_face.embedding
    emb = emb / np.linalg.norm(emb)
    return [emb.tolist()]

# -------------------------
# API Endpoints
# -------------------------
@app.post("/analyze")
async def analyze_endpoint(files: List[UploadFile] = File(...)):
    """
    Step 1: Analyze images and return quality scores.
    Does NOT save anything.
    """
    results = []
    for file in files:
        contents = await file.read()
        if not contents: continue
        nparr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None: continue

        faces = face_app.get(img)
        if not faces:
            results.append({"filename": file.filename, "face_detected": False})
            continue

        # Check largest face
        largest_face = max(faces, key=lambda f: (f.bbox[2]-f.bbox[0]) * (f.bbox[3]-f.bbox[1]))
        quality_score, details = get_image_quality(img, largest_face)
        
        results.append({
            "filename": file.filename,
            "face_detected": True,
            "quality_score": float(quality_score),
            "is_trainable": bool(quality_score >= TRAINING_QUALITY_THRESHOLD),
            "details": details
        })
    
    return {"status": "success", "analysis": results}
@app.post("/train")
async def train_endpoint(
    user_name: str = Form(...),
    user_id: str = Form(...),
    hostel_id: str = Form(...),
    created_by: str = Form(...),
    role: str = Form("resident"),
    files: List[UploadFile] = File(...)
):
    """
    Train ONE person using multiple images and store in MongoDB.
    """
    if not user_name.strip():
        raise HTTPException(status_code=400, detail="User Name is required")
    if not files:
        raise HTTPException(status_code=400, detail="No images uploaded")

    trained_count = 0
    all_embeddings = []
    all_image_paths = []

    for file in files:
        contents = await file.read()
        if not contents: continue
        nparr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None: continue

        embeddings = extract_embeddings(img)
        if not embeddings: continue

        # Save image locally
        img_path = save_trained_image(user_name, img)
        all_image_paths.append(img_path)

        for emb in embeddings:
            all_embeddings.append(emb)
            trained_count += 1

    if trained_count == 0:
        raise HTTPException(status_code=400, detail="No faces detected in images")

    # Store in MongoDB
    document = {
        "hostel_id": hostel_id,
        "user_name": user_name,
        "user_id": user_id,
        "role": role,
        "saved_images_path": all_image_paths,
        "created_at": datetime.now(),
        "created_by": created_by,
        "encodings": all_embeddings
    }
    
    # Use insert_one to keep every training session as a separate record (log-style)
    # This prevents overwriting data if the same resident ID is trained multiple times
    collection_trained.insert_one(document)

    return {
        "status": "training_complete",
        "user_name": user_name,
        "role": role,
        "images_processed": len(all_image_paths),
        "total_embeddings": len(all_embeddings)
    }

if __name__ == "__main__":
    print("\n🚀 Face Trainer API running at http://localhost:8001")
    uvicorn.run(app, host="0.0.0.0", port=8001)
