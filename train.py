# train.py
import os
import json
import numpy as np
import cv2
from insightface.app import FaceAnalysis
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
import uvicorn
import argparse

# Default configurations
TRAIN_DIR = "train_images"
OUTPUT_JSON = "embeddings.json"
TRAINED_DIR = "Trained"


# FastAPI instance for the "endpoint"
app = FastAPI(title="Face Trainer Service")

# Initialize InsightFace (global to share across API/CLI)
print("Loading face analysis model...")
face_app = FaceAnalysis(name="buffalo_l", providers=['CPUExecutionProvider'])
face_app.prepare(ctx_id=-1)

def load_db():
    if os.path.exists(OUTPUT_JSON):
        try:
            with open(OUTPUT_JSON, "r") as f:
                return json.load(f)
        except:
            return {}
    return {}

def save_db(face_db):
    with open(OUTPUT_JSON, "w") as f:
        json.dump(face_db, f, indent=4)

def save_trained_image(name, image):
    """
    Save training dataset images into Trained/<name>/
    """
    person_dir = os.path.join(TRAINED_DIR, name)
    os.makedirs(person_dir, exist_ok=True)

    count = len(os.listdir(person_dir)) + 1
    path = os.path.join(person_dir, f"{count}.jpg")

    cv2.imwrite(path, image)

def get_name(filename):
    """Extract name from filename, removing extensions and trailing numbers."""
    # filename format: name1.jpg, name2.png -> name
    return filename.split('.')[0].rstrip("0123456789")

def extract_embeddings(img,):
    """
    Detect faces and return embeddings.
    """
    faces = face_app.get(img)

    if len(faces) == 0:
        return []

    embeddings = []
    for face in faces:
        emb = face.embedding
        emb = emb / np.linalg.norm(emb)
        embeddings.append(emb.tolist())

    return embeddings

# -------------------------
# API Endpoints
# -------------------------
@app.post("/train")
async def train_endpoint(
    name: str = Form(...),
    files: list[UploadFile] = File(...)
):
    """
    Train ONE person using multiple images.

    name: person's name
    files: multiple face images
    """

    if not name.strip():
        raise HTTPException(status_code=400, detail="Name is required")

    if len(files) == 0:
        raise HTTPException(status_code=400, detail="No images uploaded")

    db = load_db()
    name = name.strip()

    if name not in db:
        db[name] = []

    if not isinstance(db[name], list):
        db[name] = [db[name]]

    trained_count = 0

    for file in files:
        contents = await file.read()

        if not contents:
            continue

        nparr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if img is None:
            continue

        embeddings = extract_embeddings(img)

        if not embeddings:
            continue

        # ✅ save image to Trained folder
        save_trained_image(name, img)

        for emb in embeddings:
            db[name].append(emb)
            trained_count += 1

    if trained_count == 0:
        raise HTTPException(status_code=400, detail="No faces detected")

    save_db(db)

    return {
        "status": "training_complete",
        "person": name,
        "images_processed": trained_count,
        "total_embeddings": len(db[name])
    }

@app.post("/scan")
async def scan_folder_endpoint(folder_path: str = Form(TRAIN_DIR)):
    """API endpoint to trigger a scan of a local directory."""
    count = run_folder_training(folder_path)
    return {"status": "success", "processed_files": count}

# -------------------------
# Local processing logic
# -------------------------
def run_folder_training(train_dir):
    """Scans a directory and updates the face database."""
    if not os.path.exists(train_dir):
        print(f"Directory not found: {train_dir}")
        return 0

    face_db = load_db()
    processed_count = 0

    for file in os.listdir(train_dir):
        path = os.path.join(train_dir, file)
        if not file.lower().endswith((".png", ".jpg", ".jpeg")):
            continue

        name = get_name(file)
        img = cv2.imread(path)
        if img is None: continue

        emb_list = extract_embeddings(img)
        if emb_list:
            if name not in face_db:
                face_db[name] = []
            
            # For robustness, handle case where DB might have single embeddings
            if not isinstance(face_db[name], list):
                face_db[name] = [face_db[name]]

            # save image copy
            save_trained_image(name, img)

            for emb in emb_list:

                face_db[name].append(emb)
                processed_count += 1
            print(f"Processed: {file} -> Identity: {name}")

    save_db(face_db)
    print(f"\nTraining complete. Processed {processed_count} images.")
    return processed_count

# -------------------------
# Entry Point
# -------------------------
# -------------------------
# Entry Point (API Service)
# -------------------------
if __name__ == "__main__":
    print("\n🚀 Face Trainer API running at http://localhost:8001")
    uvicorn.run(app, host="0.0.0.0", port=8001)
