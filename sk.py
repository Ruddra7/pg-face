# sk.py
import os
import cv2
import numpy as np
import threading
import time
import recognition
from flask import Flask, Response, render_template_string
from skimage.feature import local_binary_pattern

# Configure OpenCV for RTSP
os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;udp|max_delay;0"

# Settings for snapshots
MID_CAPTURE_INTERVAL = 3.0 # Seconds between intermediate snapshots
CAPTURES_DIR = "BodyCaptures"
FULL_CAPTURES_DIR = "FullCaptures"
os.makedirs(CAPTURES_DIR, exist_ok=True)
os.makedirs(FULL_CAPTURES_DIR, exist_ok=True)

def get_next_person_id():
    """Finds the next available Person ID by scanning the captures directory."""
    max_id = -1
    for base_dir in [CAPTURES_DIR, FULL_CAPTURES_DIR]:
        if os.path.exists(base_dir):
            for dirname in os.listdir(base_dir):
                if dirname.startswith("Person_"):
                    try:
                        pid = int(dirname.split("_")[1])
                        if pid > max_id:
                            max_id = pid
                    except:
                        pass
    return max_id + 1

app = Flask(__name__)

# -------------------------
# UTILITIES
# -------------------------

def save_snapshots(frame, box, body_id, event_type):
    """
    Saves dual snapshots: a cropped version and a full-frame version.
    event_type: 'entry', 'exit', or 'mid'
    """
    timestamp = time.strftime("%Y-%m-%d_%H-%M-%S")
    filename = f"{event_type}_{timestamp}.jpg"
    
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = map(int, box)
    
    # --- 1. Save Full Frame Snapshot (with bounding box) ---
    full_person_dir = os.path.join(FULL_CAPTURES_DIR, f"Person_{body_id}")
    os.makedirs(full_person_dir, exist_ok=True)
    full_path = os.path.join(full_person_dir, filename)
    
    full_snap = frame.copy()
    cv2.rectangle(full_snap, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv2.putText(full_snap, f"Person {body_id} {event_type.upper()}", (x1, y1-10),
                cv2.FONT_HERSHEY_DUPLEX, 0.7, (0, 255, 0), 2)
    cv2.imwrite(full_path, full_snap)

    # --- 2. Save Cropped Body Snapshot ---
    crop_person_dir = os.path.join(CAPTURES_DIR, f"Person_{body_id}")
    os.makedirs(crop_person_dir, exist_ok=True)
    crop_path = os.path.join(crop_person_dir, filename)

    # Add small padding to the crop
    pad_h = int((y2 - y1) * 0.1)
    pad_w = int((x2 - x1) * 0.1)
    
    cx1, cy1 = max(0, x1 - pad_w), max(0, y1 - pad_h)
    cx2, cy2 = min(w, x2 + pad_w), min(h, y2 + pad_h)
    
    crop = frame[cy1:cy2, cx1:cx2]
    
    if crop.size != 0:
        cv2.putText(crop, f"{event_type.upper()}", (5, 20),
                    cv2.FONT_HERSHEY_DUPLEX, 0.5, (0, 255, 0), 1)
        cv2.imwrite(crop_path, crop)
        print(f"[SAVED] Snapshots for Person {body_id} ({event_type.upper()})")
    else:
        print(f"[WARN] Empty crop for Person {body_id}, skipping crop save.")

def extract_torso(frame, box):
    x1, y1, x2, y2 = map(int, box)
    h, w = frame.shape[:2]
    
    # Clip box to frame boundaries
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    
    bh = y2 - y1
    # focus on torso region (approx 20% to 60% of body height)
    ty1 = int(y1 + 0.2 * bh)
    ty2 = int(y1 + 0.6 * bh)
    
    torso = frame[ty1:ty2, x1:x2]
    if torso.size == 0:
        return None
    return torso

def torso_color_histogram(torso):
    hsv = cv2.cvtColor(torso, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1], None, [32, 32], [0, 180, 0, 256])
    cv2.normalize(hist, hist)
    return hist.flatten()

def torso_texture_descriptor(torso):
    gray = cv2.cvtColor(torso, cv2.COLOR_BGR2GRAY)
    lbp = local_binary_pattern(gray, 8, 1, method="uniform")
    hist, _ = np.histogram(lbp.ravel(), bins=np.arange(0, 10), range=(0, 9))
    hist = hist.astype("float")
    hist /= (hist.sum() + 1e-6)
    return hist

def compute_appearance_signature(frame, box):
    torso = extract_torso(frame, box)
    if torso is None: return None
    color_hist = torso_color_histogram(torso)
    texture_hist = torso_texture_descriptor(torso)
    return np.concatenate([color_hist, texture_hist])

def appearance_similarity(sig1, sig2):
    if sig1 is None or sig2 is None: return 0
    return float(np.dot(sig1, sig2) / (np.linalg.norm(sig1) * np.linalg.norm(sig2) + 1e-6))

def box_iou(a, b):
    xA, yA = max(a[0], b[0]), max(a[1], b[1])
    xB, yB = min(a[2], b[2]), min(a[3], b[3])
    if xB <= xA or yB <= yA: return 0
    inter = (xB - xA) * (yB - yA)
    areaA = (a[2]-a[0]) * (a[3]-a[1])
    areaB = (b[2]-b[0]) * (b[3]-b[1])
    return inter / float(areaA + areaB - inter)

# -------------------------
# LOAD MODELS (MobileNetSSD for Person Detection)
# -------------------------
print("Loading body detector...")
# Ensure these files exist in the same directory as sk.py
person_net = cv2.dnn.readNetFromCaffe(
    "MobileNetSSD_deploy.prototxt",
    "MobileNetSSD_deploy.caffemodel"
)
PERSON_CLASS_ID = 15

# -------------------------
# SHARED STATE
# -------------------------
class SharedState:
    def __init__(self):
        self.current_frame = None
        self.display_frame = None
        self.lock = threading.Lock()
        self.running = True
        self.body_tracks = {}
        self.session_data = {}
        self.next_body_id = get_next_person_id()
        self.lost_tracks = {}
        self.last_bodies = []

state = SharedState()

def new_session(frame, box, body_id):
    """Initializes a tracking session and captures entry image."""
    now = time.time()
    session = {
        "start_time": now,
        "last_seen": now,
        "last_mid_capture": now,
        "finalized": False,
        "last_body_box": box,
        "last_frame": frame.copy(),
        "body_id": body_id,
        "entry_saved": False,
        "exit_saved": False
    }
    # Save Snapshots immediately
    save_snapshots(frame, box, body_id, "entry")
    session["entry_saved"] = True
    return session

# -------------------------
# CAMERA THREAD
# -------------------------
def camera_thread():
    RTSP_URL = "rtsp://admin:vipras@4455@192.168.0.121:554/cam/realmonitor?channel=1&subtype=0"
    cap = cv2.VideoCapture(RTSP_URL, cv2.CAP_FFMPEG)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    while state.running:
        # Buffer clearing: grab a few frames and discard
        for _ in range(3): cap.grab()
        ret, frame = cap.read()
        if not ret:
            print("Reconnecting stream...")
            cap.release()
            time.sleep(1.0)
            cap = cv2.VideoCapture(RTSP_URL, cv2.CAP_FFMPEG)
            continue
            
        frame = cv2.resize(frame, (640, 480))
        # mirror view (lateral inversion) as requested by the original code's style
        frame = cv2.flip(frame, 1) 
        
        with state.lock:
            state.current_frame = frame
            state.display_frame = frame

# -------------------------
# BODY DETECTION & TRACKING
# -------------------------
def detect_bodies(frame):
    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 0.007843, (300, 300), 127.5)
    person_net.setInput(blob)
    detections = person_net.forward()
    bodies = []
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        class_id = int(detections[0, 0, i, 1])
        if class_id == PERSON_CLASS_ID and confidence > 0.6:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            bodies.append(box.astype(int))
    return bodies

def match_body(box, current_frame):
    best_id = None
    best_score = 0
    new_center = ((box[0] + box[2]) // 2, (box[1] + box[3]) // 2)

    # 1. Positional Sticky Lock (if person moved very little)
    STICKY_WINDOW = 2.0
    close_candidates = []
    for existing_id, data in state.body_tracks.items():
        if time.time() - data["last_seen"] > STICKY_WINDOW: continue
        prev_center = data.get("center")
        if prev_center:
            dist = np.linalg.norm(np.array(prev_center) - np.array(new_center))
            if dist < 120: close_candidates.append((existing_id, dist))

    if len(close_candidates) == 1:
        bid = close_candidates[0][0]
        state.body_tracks[bid].update({
            "box": box, 
            "last_seen": time.time(), 
            "prev_center": state.body_tracks[bid].get("center"), 
            "center": new_center
        })
        return bid

    # 2. IOU Match (overlap)
    for existing_id, data in state.body_tracks.items():
        if time.time() - data["last_seen"] > 2.0: continue
        iou = box_iou(box, data["box"])
        if iou > best_score:
            best_score = iou
            best_id = existing_id

    # 3. Appearance Similarity (fallback)
    new_sig = compute_appearance_signature(current_frame, box)
    if best_score < 0.45:
        for existing_id, data in state.body_tracks.items():
            if time.time() - data["last_seen"] > 2.0: continue
            sim = appearance_similarity(data.get("appearance"), new_sig)
            if sim > 0.90:
                best_id = existing_id
                break

    if best_id is not None:
        data = state.body_tracks[best_id]
        data.update({
            "box": box, 
            "last_seen": time.time(), 
            "center": new_center, 
            "appearance": (0.8 * data["appearance"] + 0.2 * new_sig) if data["appearance"] is not None else new_sig
        })
        return best_id

    # 4. Reconnect to lost tracks (using appearance)
    for lost_id, data in list(state.lost_tracks.items()):
        if time.time() - data["last_seen"] > 8.0:
            state.lost_tracks.pop(lost_id, None)
            continue
        if new_sig is not None and appearance_similarity(data.get("appearance"), new_sig) > 0.75:
            state.body_tracks[lost_id] = data
            state.body_tracks[lost_id].update({"box": box, "last_seen": time.time(), "center": new_center})
            return lost_id

    # 5. Create new track
    bid = state.next_body_id
    state.next_body_id += 1
    state.body_tracks[bid] = {
        "box": box,
        "last_seen": time.time(),
        "appearance": new_sig,
        "center": new_center,
        "prev_center": None
    }
    return bid

# -------------------------
# DETECTION THREAD
# -------------------------
def detection_thread():
    last_detect_time = 0
    DETECT_INTERVAL = 0.4  # Run detection 2.5 times per second
    PERSISTENCE_WINDOW = 5.0 # Seconds before considering a track exited

    while state.running:
        with state.lock:
            frame = None if state.current_frame is None else state.current_frame.copy()
        
        if frame is None:
            time.sleep(0.01)
            continue

        now = time.time()
        if now - last_detect_time > DETECT_INTERVAL:
            bodies = detect_bodies(frame)
            state.last_bodies = bodies
            last_detect_time = now

            for body_box in bodies:
                bid = match_body(body_box, frame)
                if bid not in state.session_data:
                    state.session_data[bid] = new_session(frame, body_box, bid)
                else:
                    session = state.session_data[bid]
                    session["last_seen"] = now
                    session["last_body_box"] = body_box
                    session["last_frame"] = frame.copy()
                    
                    # Periodic mid-session captures
                    if now - session.get("last_mid_capture", 0) > MID_CAPTURE_INTERVAL:
                        save_snapshots(frame, body_box, bid, "mid")
                        session["last_mid_capture"] = now

        # Handle Exits (when a track hasn't been seen for PERSISTENCE_WINDOW)
        for bid in list(state.body_tracks.keys()):
            body = state.body_tracks[bid]
            if time.time() - body["last_seen"] > PERSISTENCE_WINDOW:
                session = state.session_data.get(bid)
                if session and not session["exit_saved"]:
                    save_snapshots(session["last_frame"], session["last_body_box"], bid, "exit")
                    session["exit_saved"] = True
                    session["finalized"] = True
                    
                    # Trigger separate recognition process
                    threading.Thread(target=recognition.process_person_exit, args=(bid,), daemon=True).start()
                
                with state.lock:
                    state.lost_tracks[bid] = body
                    state.body_tracks.pop(bid, None)

        time.sleep(0.05)

# -------------------------
# STREAM & FLASK APP
# -------------------------
def generate_frames():
    """Generates MJPEG stream for the browser."""
    while True:
        with state.lock:
            frame = None if state.display_frame is None else state.display_frame.copy()
            bodies = state.body_tracks.copy()
            
        if frame is None:
            time.sleep(0.01)
            continue
            
        # Draw tracking boxes on display frame
        for bid, body in bodies.items():
            if time.time() - body["last_seen"] > 1.5: continue # Skip stale tracks for viz
            x1, y1, x2, y2 = map(int, body["box"])
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"Person {bid}", (x1, y1 - 10), 
                        cv2.FONT_HERSHEY_DUPLEX, 0.6, (0, 255, 0), 2)
            
        ret, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
        if not ret: continue
        
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
        time.sleep(0.04) # Approx 25 FPS display

@app.route('/')
def index():
    return render_template_string("""
    <html>
        <head>
            <title>Human Tracking System</title>
            <style>
                body { background: #111; color: #eee; text-align: center; font-family: sans-serif; }
                .container { margin-top: 50px; }
                img { border: 2px solid #333; border-radius: 8px; box-shadow: 0 0 20px rgba(0,255,0,0.1); }
            </style>
        </head>
        <body>
            <div class="container">
                <h2>HUMAN BODY TRACKING LIVE FEED</h2>
                <img src="/video_feed" width="80%">
                <p>Capturing entry and exit snapshots in the 'BodyCaptures' folder.</p>
            </div>
        </body>
    </html>
    """)

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__=="__main__":
    print(f"--- STARTING CAMERA SYSTEM FOR HOSTEL: {recognition.CAMERA_HOSTEL_ID} ---")
    # Start background threads
    threading.Thread(target=camera_thread, daemon=True).start()
    threading.Thread(target=detection_thread, daemon=True).start()
    
    # Start Flask app
    app.run(host='0.0.0.0', port=8000, threaded=True)