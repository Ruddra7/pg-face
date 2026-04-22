import os
import cv2
import json
import numpy as np
import time
import requests
from datetime import datetime
from insightface.app import FaceAnalysis
from pymongo import MongoClient
from s3 import upload_to_s3

# MongoDB Setup
MONGO_URI = "mongodb+srv://faceerp:faceerp@cluster0.kvd3q8j.mongodb.net/erp?retryWrites=true&w=majority&appName=Cluster0"
client = MongoClient(MONGO_URI)
db_mongo = client["pg_axis_erp"]
collection_trained = db_mongo["resident_trained_data"]
collection_result = db_mongo["recognition_result"]

# Constants
RESULTS_PATH = "recognition_results.json"
CAPTURES_DIR = "BodyCaptures"
FULL_CAPTURES_DIR = "FullCaptures"
MATCH_THRESHOLD = 0.4 # Slightly increased for better precision
NOTIFY_SERVER_URL = "http://localhost:8002/api/recognition/notify" # Adjust to your server project URL

# Camera-specific configuration (The hostel where this physical camera is installed)
# Use an environment variable 'CAMERA_HOSTEL_ID' or fallback to a default ID
CAMERA_HOSTEL_ID = os.getenv("CAMERA_HOSTEL_ID", "69d8d59959e0bda224a628a5")

# Load Face Analysis
print("[RECOGNITION] Initializing face models...")
face_app = FaceAnalysis(name="buffalo_l", providers=['CPUExecutionProvider'])
face_app.prepare(ctx_id=-1)
rec_model = face_app.models['recognition']

def load_trained_data():
    """Loads all resident embeddings from MongoDB, using document _id for uniqueness."""
    docs = list(collection_trained.find({}))
    trained_map = {} # doc_id_str -> {'name': name, 'encodings': [], 'resident_id': rid}
    
    for doc in docs:
        doc_id = str(doc.get('_id'))
        # Support both new 'user_id' and old 'resident_id'
        uid = doc.get('user_id') or doc.get('resident_id')
        name = doc.get('user_name') or doc.get('resident_name')
        encodings = doc.get('encodings', [])
        
        if not doc_id: continue
        
        # Using doc_id as the key ensures that even if residentid is repeated,
        # each document's embeddings are considered separately.
        trained_map[doc_id] = {
            "name": name,
            "user_id": uid,
            "role": doc.get('role', 'resident'),
            "encodings": encodings,
            "hostel_id": doc.get('hostel_id')
        }
        
    return trained_map

def normalize(v):
    norm = np.linalg.norm(v)
    if norm == 0: return v
    return v / norm

def process_person_exit(body_id):
    """
    Analyzes all captured images for a person and stores results in MongoDB & JSON.
    """
    person_dir = os.path.join(CAPTURES_DIR, f"Person_{body_id}")
    full_person_dir = os.path.join(FULL_CAPTURES_DIR, f"Person_{body_id}")
    
    if not os.path.exists(person_dir):
        print(f"[RECOGNITION] No capture directory for Person {body_id}")
        return

    trained_db = load_trained_data()
    images = [f for f in os.listdir(person_dir) if f.endswith('.jpg')]
    if not images: return

    print(f"[RECOGNITION] Analyzing Person {body_id} ({len(images)} frames)...")
    
    from insightface.app.common import Face

    best_overall_score = 0
    best_match_id = None
    best_match_name = None
    best_match_role = None
    
    total_detections = 0
    person_scores = {} # doc_id -> max similarity score seen
    
    valid_face_frames = [] # list of (score, img_path, img_data)

    for img_name in images:
        path = os.path.join(person_dir, img_name)
        img = cv2.imread(path)
        if img is None: continue
        
        # Detect face
        bboxes, kpss = face_app.det_model.detect(img, max_num=1) 
        
        if bboxes is not None and len(bboxes) > 0:
            det_score = bboxes[0][4]
            # Only consider reasonably clear faces
            if det_score > 0.45:
                total_detections += 1
                bbox = bboxes[0][:4]
                kps = kpss[0]
                
                face_obj = Face(bbox=bbox, kps=kps, det_score=det_score)
                rec_model.get(img, face_obj)
                
                if face_obj.embedding is not None:
                    emb = normalize(face_obj.embedding)
                    
                    # Store as a candidate for "best clear face"
                    valid_face_frames.append({
                        "det_score": det_score,
                        "path": path,
                        "img": img
                    })

                    # Compare against MongoDB residents
                    for rid, data in trained_db.items():
                        vectors = np.array(data['encodings'])
                        if vectors.size == 0: continue
                        
                        # Calculate dot products for all stored embeddings of this resident
                        sims = np.dot(vectors, emb)
                        max_s = float(np.max(sims))
                        
                        # Track the best score ever seen for this resident
                        if max_s > person_scores.get(rid, 0):
                            person_scores[rid] = max_s

    # Determining Final Winner
    best_match_hostel_id = None
    if person_scores:
        # Find the winner doc_id
        winner_doc_id = max(person_scores, key=person_scores.get)
        winner_score = person_scores[winner_doc_id]
        
        if winner_score > MATCH_THRESHOLD:
            best_match_id = trained_db[winner_doc_id]['user_id']
            best_match_name = trained_db[winner_doc_id]['name']
            best_match_role = trained_db[winner_doc_id].get('role', 'resident')
            best_match_hostel_id = trained_db[winner_doc_id].get('hostel_id')
            best_overall_score = winner_score

    # --- Classification Logic ---
    status = "Unidentified"
    if best_match_id:
        status = "Authorized"
    elif total_detections > 0:
        status = "Unauthorized"
    else:
        status = "Unidentified"

    # --- Cropped Image Selection Logic ---
    final_crops = []
    if status in ["Authorized", "Unauthorized"]:
        # Store only ONE best clear face image
        if valid_face_frames:
            # Sort by detection score to get the clearest one
            valid_face_frames.sort(key=lambda x: x['det_score'], reverse=True)
            final_crops.append(valid_face_frames[0]['path'])
    else:
        # Unidentified: Store ALL cropped images
        final_crops = [os.path.join(person_dir, f) for f in images]

    # --- S3 Upload Logic ---
    s3_full_frame_urls = []
    if os.path.exists(full_person_dir):
        full_imgs = [f for f in os.listdir(full_person_dir) if f.endswith('.jpg')]
        for img_name in full_imgs:
            local_path = os.path.join(full_person_dir, img_name)
            s3_url = upload_to_s3(local_path, f"full_frames/Person_{body_id}")
            if s3_url: s3_full_frame_urls.append(s3_url)

    s3_crop_urls = []
    for crop_path in final_crops:
        s3_url = upload_to_s3(crop_path, f"crops/Person_{body_id}")
        if s3_url: s3_crop_urls.append(s3_url)

    # --- Prepare Final Record ---
    try:
        pid = int(body_id)
    except:
        pid = body_id

    result = {
        "person_id": pid,
        "user_id": best_match_id,
        "user_name": best_match_name,
        "role": best_match_role if best_match_id else None,
        "status": status,
        "confidence": round(float(best_overall_score), 3),
        "full_frame_urls": s3_full_frame_urls,
        "cropped_image_urls": s3_crop_urls,
        "hostel_id": CAMERA_HOSTEL_ID, # Always log the location of the camera
        "resident_home_hostel_id": best_match_hostel_id, # Optional: track their home hostel
        "timestamp": datetime.now()
    }
    
    # Save to MongoDB
    collection_result.insert_one(result)
    
    # Send Signal to Server Project (Minimal signal, no data)
    try:
        payload = {"event": "RECOGNITION_COMPLETE"}
        if result.get("hostel_id"):
            payload["hostel_id"] = result["hostel_id"]
        requests.post(NOTIFY_SERVER_URL, json=payload, timeout=2)
    except Exception as e:
        print(f"[RECOGNITION] Failed to notify server: {e}")
    
    # Save to JSON (making it JSON serializable)
    json_record = result.copy()
    json_record["timestamp"] = json_record["timestamp"].strftime("%Y-%m-%d %H:%M:%S")
    del json_record["_id"]
    save_to_json(json_record)

    print(f"[RECOGNITION] Body {body_id} -> {status} ({best_match_name or 'N/A'})")

def save_to_json(data):
    results = []
    if os.path.exists(RESULTS_PATH):
        try:
            with open(RESULTS_PATH, "r") as f:
                results = json.load(f)
        except:
            results = []
    results.append(data)
    with open(RESULTS_PATH, "w") as f:
        json.dump(results, f, indent=4)

if __name__ == "__main__":
    print("Recognition module (MongoDB) ready.")
