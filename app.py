# app.py
import os
os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;udp|max_delay;0"
import cv2
import json
import numpy as np
import threading
import time
import queue
from flask import Flask, Response, render_template_string, session
from insightface.app import FaceAnalysis
from skimage.feature import local_binary_pattern

try:
    from insightface.app.common import Face
except ImportError:
    class Face:
        def __init__(self, bbox, kps, det_score, embedding=None):
            self.bbox = bbox
            self.kps = kps
            self.det_score = det_score
            self.embedding = embedding

DB_PATH = "embeddings.json"

AUTHORIZED_DIR = "Authorized"
UNAUTHORIZED_DIR = "Unauthorized"
OBSTRUCTED_DIR = "Unidentified"
AUTHORIZED_THRESHOLD = 0.55
UNKNOWN_THRESHOLD = 0.45
MAX_ATTEMPTS = 5

MIN_FACE_SIZE = 65

app = Flask(__name__)

# -------------------------
# UTILITIES
# -------------------------

def load_db():
    try:
        with open(DB_PATH, "r") as f:
            return json.load(f)
    except:
        return {}

def normalize(v):
    return v / np.linalg.norm(v)

def save_snapshot(frame, box, label, folder, color):
    os.makedirs(folder, exist_ok=True)
    filename = time.strftime("%Y-%m-%d_%H-%M-%S") + ".jpg"
    path = os.path.join(folder, filename)

    snap = frame.copy()
    cv2.rectangle(snap, (box[0], box[1]), (box[2], box[3]), color, 3)
    cv2.putText(snap, label, (box[0], box[1]-10),
                cv2.FONT_HERSHEY_DUPLEX, 0.8, color, 2)
    cv2.imwrite(path, snap)

def face_is_obstructed(kps, box):
    if kps is None:
        return True

    left_eye, right_eye, nose, left_mouth, right_mouth = kps
    face_width = box[2] - box[0]

    eye_distance = np.linalg.norm(left_eye - right_eye)

    if eye_distance < face_width * 0.15:
        return True

    mid_x = (left_eye[0] + right_eye[0]) / 2
    if abs(nose[0] - mid_x) > face_width * 0.25:
        return True

    if np.linalg.norm(left_mouth - right_mouth) < face_width * 0.12:
        return True

    return False

def recognize_face(embedding, db):
    best_match = None
    best_score = 0

    for name, vectors in db.items():
        vectors = np.array(vectors)
        score = float(np.max(np.dot(vectors, embedding)))

        if score < 0.50:
            continue

        if score > best_score:
            best_score = score
            best_match = name

    return best_match, best_score

def extract_torso(frame, box):
    x1, y1, x2, y2 = map(int, box)
    h = y2 - y1

    # focus on torso region
    ty1 = int(y1 + 0.2 * h)
    ty2 = int(y1 + 0.6 * h)

    torso = frame[ty1:ty2, x1:x2]

    if torso.size == 0:
        return None

    return torso

def torso_color_histogram(torso):
    hsv = cv2.cvtColor(torso, cv2.COLOR_BGR2HSV)

    hist = cv2.calcHist(
        [hsv],
        [0, 1],
        None,
        [32, 32],
        [0, 180, 0, 256]
    )

    cv2.normalize(hist, hist)
    return hist.flatten()

from skimage.feature import local_binary_pattern

def torso_texture_descriptor(torso):
    gray = cv2.cvtColor(torso, cv2.COLOR_BGR2GRAY)

    lbp = local_binary_pattern(gray, 8, 1, method="uniform")

    hist, _ = np.histogram(
        lbp.ravel(),
        bins=np.arange(0, 10),
        range=(0, 9)
    )

    hist = hist.astype("float")
    hist /= (hist.sum() + 1e-6)

    return hist

def compute_appearance_signature(frame, box):
    torso = extract_torso(frame, box)

    if torso is None:
        return None

    color_hist = torso_color_histogram(torso)
    texture_hist = torso_texture_descriptor(torso)

    return np.concatenate([color_hist, texture_hist])

def appearance_similarity(sig1, sig2):
    if sig1 is None or sig2 is None:
        return 0

    return float(np.dot(sig1, sig2) / (
        np.linalg.norm(sig1) * np.linalg.norm(sig2) + 1e-6
    ))

# -------------------------
# LOAD MODELS
# -------------------------
print("Loading models...")
face_app = FaceAnalysis(name="buffalo_l", providers=['CPUExecutionProvider'])
face_app.prepare(ctx_id=-1)
det_model = face_app.det_model
rec_model = face_app.models['recognition']

print("Loading body detector...")

person_net = cv2.dnn.readNetFromCaffe(
    "MobileNetSSD_deploy.prototxt",
    "MobileNetSSD_deploy.caffemodel"
)

PERSON_CLASS_ID = 15

face_db = load_db()

def new_session(frame, box, body_id):
    return {
        "best_score": 0,
        "attempts": 0,
        "obstructed": 0,
        "clear": 0,
        "start_time": time.time(),
        "last_seen": time.time(),
        "finalized": False,
        "last_box": box,
        "last_frame": frame.copy(),
        "body_id": body_id,
        "identity_votes": 0,
        "identity": None,
        "exit_time": None,
        "cooldown_until": 0,
        "snapshot_saved": False,
        "unidentified_saved": False,
        "unknown_saved": False,
        "authorized_saved": False,
        "authorized_confirmed": False
    }

# -------------------------
# SHARED STATE
# -------------------------
class SharedState:
    def __init__(self):
        self.current_frame = None
        self.display_frame = None
        self.detections = []
        self.lock = threading.Lock()
        self.running = True
        self.rec_queue = queue.Queue(maxsize=1)

        self.last_bodies = []

        self.name_cache = {}
        self.saved_ids = set()

        self.session_data = {}

        # NEW: identity decision tracking
        self.best_scores = {}
        self.attempt_counts = {}
        self.finalized_ids = set()

        self.body_tracks = {}
        self.face_to_body = {}
        self.next_body_id = 0
        self.lost_tracks = {}

state = SharedState()

# -------------------------
# CAMERA THREAD
# -------------------------
def camera_thread():
    RTSP_URL = "rtsp://admin:admin123@10.126.150.17:554/ch0_0.264"

    cap = cv2.VideoCapture(RTSP_URL, cv2.CAP_FFMPEG)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    while state.running:

        # 🔥 grab newest frame only
        for _ in range(3):
            cap.grab()

        ret, frame = cap.read()

        if not ret:
            print("Reconnecting stream...")
            cap.release()
            time.sleep(0.3)
            cap = cv2.VideoCapture(RTSP_URL, cv2.CAP_FFMPEG)
            continue

        frame = cv2.resize(frame, (640, 480))

        # mirror view (lateral inversion)
        frame = cv2.flip(frame, 1)

        with state.lock:
            state.current_frame = frame
            state.display_frame = frame

# -------------------------
# RECOGNITION WORKER
# -------------------------
def recognition_worker():
    while state.running:
        try:
            frame, face_obj, fid, box = state.rec_queue.get(timeout=1)

            rec_model.get(frame, face_obj)

            if face_obj.embedding is not None:

                name, score = recognize_face(
                    normalize(face_obj.embedding),
                    face_db
                )

                session = state.session_data.get(fid)
                if not session:
                    state.rec_queue.task_done()
                    continue

                session.setdefault("attempts", 0)
                session.setdefault("best_score", 0)

                session["attempts"] += 1

                if score > session["best_score"]:
                    session["best_score"] = score
                    session["best_identity"] = name or "Unknown"

                    # ⭐ store best frame & face box
                    session["best_frame"] = frame.copy()
                    session["best_face_box"] = box.copy()

                with state.lock:

                    # update best score seen so far
                    prev_best = state.best_scores.get(fid, 0)
                    if score > prev_best:
                        state.best_scores[fid] = score

                    # increment attempts
                    state.attempt_counts[fid] = state.attempt_counts.get(fid, 0) + 1

                    # ✅ IDENTITY DECISION LOGIC
                    if session:

                        # ✔ strong match → vote
                        if score >= AUTHORIZED_THRESHOLD and name != "Unknown":
                            session["identity_votes"] = session.get("identity_votes", 0) + 1

                            if session["identity_votes"] >= 1:
                                session["identity"] = name
                                session["identity_locked"] = True
                                session["authorized_confirmed"] = True
                              
                        # ✔ after enough attempts mark temporary unknown
                        elif session["attempts"] >= 6 and session.get("identity") is None:
                            session["identity"] = "UNKNOWN"   # temporary only

                        if fid not in state.saved_ids:
                            state.saved_ids.add(fid)

            state.rec_queue.task_done()

        except queue.Empty:
            continue

def box_iou(a, b):
    xA = max(a[0], b[0])
    yA = max(a[1], b[1])
    xB = min(a[2], b[2])
    yB = min(a[3], b[3])

    if xB <= xA or yB <= yA:
        return 0

    inter = (xB - xA) * (yB - yA)
    areaA = (a[2]-a[0]) * (a[3]-a[1])
    areaB = (b[2]-b[0]) * (b[3]-b[1])

    return inter / float(areaA + areaB - inter)

def detect_bodies(frame):
    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(
        cv2.resize(frame, (300, 300)),
        0.007843,
        (300, 300),
        127.5
    )

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

def match_body(box, state):
    best_id = None
    best_score = 0

    # 🔥 SMART STICKY LOCK (only if single clear candidate)
    STICKY_WINDOW = 2
    new_center = ((box[0] + box[2]) // 2, (box[1] + box[3]) // 2)

    close_candidates = []

    for existing_id, data in state.body_tracks.items():

        if time.time() - data["last_seen"] > STICKY_WINDOW:
            continue

        prev_center = data.get("center")
        if prev_center is None:
            continue

        dist = np.linalg.norm(np.array(prev_center) - np.array(new_center))

        if dist < 120:   # reduced threshold
            close_candidates.append((existing_id, dist))

    # Only lock if exactly ONE candidate
    if len(close_candidates) == 1:
        existing_id = close_candidates[0][0]
        data = state.body_tracks[existing_id]

        data["box"] = box
        data["last_seen"] = time.time()
        data["prev_center"] = data.get("center")
        data["center"] = new_center

        return existing_id

    for existing_id, data in state.body_tracks.items():

        # 🔒 Do not reconnect locked identities
        session = state.session_data.get(existing_id)
        if session and session.get("identity_locked"):
            continue

        if time.time() - data["last_seen"] > 2:
            continue

        prev = data["box"]
        iou = box_iou(box, prev)

        if iou > best_score:
            best_score = iou
            best_id = existing_id

    # try appearance-based reconnect
    new_sig = compute_appearance_signature(state.current_frame, box)

    for existing_id, data in state.body_tracks.items():

        # only consider recently lost tracks
        if time.time() - data["last_seen"] > 2:
            continue

        prev_sig = data.get("appearance")

        if prev_sig is None or new_sig is None:
            continue

        sim = appearance_similarity(prev_sig, new_sig)

        if sim > 0.92:
            data["box"] = box
            data["last_seen"] = time.time()

            cx = (box[0] + box[2]) // 2
            cy = (box[1] + box[3]) // 2

            data["prev_center"] = data.get("center")
            data["center"] = (cx, cy)

            return existing_id

    if best_score > 0.45 and best_id is not None:

        session = state.session_data.get(best_id)

        # 🚫 NEVER allow reassignment of locked identities
        if session and session.get("identity_locked"):
            return None  # force creation of new track

        data = state.body_tracks[best_id]

        # 🛑 Prevent large jump reassignment
        prev_center = data.get("center")
        new_center = ((box[0]+box[2])//2, (box[1]+box[3])//2)

        if prev_center:
            dist = np.linalg.norm(np.array(prev_center) - np.array(new_center))
            if dist > 180:   # tune if needed
                best_id = None
            else:
                data["box"] = box
                data["last_seen"] = time.time()
                data["prev_center"] = prev_center
                data["center"] = new_center
                return best_id

        else:
            data["box"] = box
            data["last_seen"] = time.time()
            data["center"] = new_center
            return best_id
    
    # try reconnect to recently lost tracks
    for lost_id, data in list(state.lost_tracks.items()):

        if time.time() - data["last_seen"] > 8:
            state.lost_tracks.pop(lost_id)
            continue

        prev_sig = data.get("appearance")
        if prev_sig is None or new_sig is None:
            continue

        sim = appearance_similarity(prev_sig, new_sig)

        if sim > 0.65:
            state.body_tracks[lost_id] = data
            state.body_tracks[lost_id]["box"] = box
            state.body_tracks[lost_id]["last_seen"] = time.time()
            return lost_id
        
    # 🔥 EXIT REAPPEAR SUPPRESSION
    EXIT_REENTRY_WINDOW = 5  # seconds
    EDGE_MARGIN = 60

    new_center = ((box[0] + box[2]) // 2, (box[1] + box[3]) // 2)

    for old_bid, session in state.session_data.items():

        if not session.get("finalized"):
            continue

        if time.time() - session.get("exit_time", 0) > EXIT_REENTRY_WINDOW:
            continue

        last_box = session.get("last_body_box")
        if last_box is None:
            continue

        last_center = (
            (last_box[0] + last_box[2]) // 2,
            (last_box[1] + last_box[3]) // 2
        )

        dist = np.linalg.norm(np.array(last_center) - np.array(new_center))

        if dist < 150:
            # ignore ghost re-detection
            return old_bid
        
    # 🔥 DUPLICATE TRACK SUPPRESSION
    for existing_id, data in state.body_tracks.items():

        existing_box = data["box"]

        overlap = box_iou(box, existing_box)

        if overlap > 0.6:
            # Strong overlap → same person
            data["box"] = box
            data["last_seen"] = time.time()

            cx = (box[0] + box[2]) // 2
            cy = (box[1] + box[3]) // 2

            data["prev_center"] = data.get("center")
            data["center"] = (cx, cy)

            return existing_id

    # create new body
    bid = state.next_body_id
    state.next_body_id += 1

    cx = (box[0] + box[2]) // 2
    cy = (box[1] + box[3]) // 2

    state.body_tracks[bid] = {
        "box": box,
        "last_seen": time.time(),
        "appearance": None,
        "center": (cx, cy),
        "prev_center": None
    }

    return bid

# -------------------------
# DETECTION THREAD
# -------------------------
def detection_thread():
    tracked = []
    next_id = 0

    det_size = 640
    scale_x = 640 / 512
    scale_y = 480 / 512

    last_detect_time = 0
    DETECT_INTERVAL = 0.45   # ≈ 8 FPS detection

    last_face_time = 0
    FACE_INTERVAL = 0.50   # face detect ~5 FPS (huge CPU relief)

    while state.running:

        with state.lock:
            frame = None if state.current_frame is None else state.current_frame.copy()
            
        if frame is None:
            time.sleep(0.01)
            continue

        now_time = time.time()

        if now_time - last_detect_time > DETECT_INTERVAL:
            state.last_bodies = detect_bodies(frame)
            last_detect_time = now_time

        bodies = getattr(state, "last_bodies", [])

        for body_box in bodies:
            bid = match_body(body_box, state)

            if bid not in state.session_data:
                # 🔥 Do not create new session if recently finalized
                old_session = state.session_data.get(bid)

                if old_session and old_session.get("finalized", False):
                    # allow reconnect without resetting session
                    old_session["finalized"] = False
                    old_session["exit_time"] = None
                    old_session["cooldown_until"] = 0
                else:
                    state.session_data[bid] = new_session(frame, body_box, bid)
            else:
                state.session_data[bid]["last_seen"] = time.time()

            # 🔥 UPDATE TORSO APPEARANCE EVERY BODY DETECTION
            sig = compute_appearance_signature(frame, body_box)
            if sig is not None:
                track = state.body_tracks.get(bid)
                if track is None:
                    continue

                prev = track.get("appearance")
                if prev is None:
                    state.body_tracks[bid]["appearance"] = sig
                else:
                    state.body_tracks[bid]["appearance"] = 0.8 * prev + 0.2 * sig

        small = cv2.resize(frame,(512,512))

        if now_time - last_face_time > FACE_INTERVAL:
            bboxes, kpss = det_model.detect(small)
            last_face_time = now_time
        else:
            bboxes, kpss = None, None

        detections=[]
        now=time.time()
        
        BODY_TIMEOUT = 3.5

        if bboxes is not None:
            for i in range(bboxes.shape[0]):
                bbox=bboxes[i,:4]
                score=bboxes[i,4]
                if score<0.5: continue

                box=(bbox*[scale_x,scale_y,scale_x,scale_y]).astype(int)
                center = ((box[0]+box[2])/2, (box[1]+box[3])/2)
                if box[3]-box[1] < MIN_FACE_SIZE:
                    continue

                closest_body = None   # ← prevents UnboundLocalError

                fid = None

                for t in tracked:
                    tc=((t['box'][0]+t['box'][2])/2,(t['box'][1]+t['box'][3])/2)

                    iou = box_iou(box, t['box'])
                    dist = np.linalg.norm(np.array(center)-np.array(tc))
                    size_diff = abs((box[2]-box[0]) - (t['box'][2]-t['box'][0]))

                    if iou > 0.15 or (dist < 200 and size_diff < 120):
                        fid = t['id']

                        # 🔥 smoothing to prevent ID jitter
                        t['box'] = [
                            int(0.7*t['box'][0] + 0.3*box[0]),
                            int(0.7*t['box'][1] + 0.3*box[1]),
                            int(0.7*t['box'][2] + 0.3*box[2]),
                            int(0.7*t['box'][3] + 0.3*box[3]),
                        ]

                        t['last'] = now
                        break

                # determine closest body for this face
                face_center = ((box[0]+box[2])//2, box[3])  # use feet position

                min_dist = 9999

                for existing_id, data in state.body_tracks.items():
                    bx = data["box"]
                    body_center = ((bx[0]+bx[2])//2, (bx[1]+bx[3])//2)

                    dist = np.linalg.norm(np.array(face_center) - np.array(body_center))

                    if dist < min_dist and dist < 140:
                        min_dist = dist
                        closest_body = existing_id                        

                # get body session
                if closest_body is None:
                    continue

                session = state.session_data.get(closest_body)
                if session is None:
                    continue

                # update appearance signature
                sig = compute_appearance_signature(frame, box)

                if sig is not None:
                    prev_sig = state.body_tracks[closest_body].get("appearance")

                    if prev_sig is None:
                        state.body_tracks[closest_body]["appearance"] = sig
                    else:
                        # smooth update to stabilize against lighting change
                        state.body_tracks[closest_body]["appearance"] = \
                            0.7 * prev_sig + 0.3 * sig

                session["last_face_box"] = [int(x) for x in box]
                session["last_frame"] = frame.copy()
                session["last_face_seen"] = time.time()

                if session.get("identity"):
                    name = session["identity"]
                elif session.get("attempts", 0) < 6:
                    name = "DETECTING"
                else:
                    name = "UNKNOWN"

                # obstruction check
                obstructed = face_is_obstructed(kpss[i] if kpss is not None else None, box)

                if obstructed:
                    session["obstructed"] += 1
                    session["ever_obstructed"] = True
                else:
                    session["clear"] += 1

                    # send face for recognition
                    if not session.get("identity_locked", False) and not state.rec_queue.full():
                        kps = kpss[i].copy() if kpss is not None else None
                        if kps is not None:
                            kps[:,0] *= scale_x
                            kps[:,1] *= scale_y
                            face_obj = Face(bbox=box.astype(float), kps=kps, det_score=score)
                            state.rec_queue.put((frame.copy(), face_obj, closest_body, box))

        TRACK_TIMEOUT = 5  # seconds

        tracked=[t for t in tracked if now-t['last'] < TRACK_TIMEOUT]

        for bid, body in state.body_tracks.items():
            if bid in state.session_data:
                session = state.session_data[bid]

        # -------- SMART EXIT CHECK --------
        FRAME_W = 640
        FRAME_H = 480
        EDGE_MARGIN = 40
        EXIT_TIMEOUT = 8

        for bid in list(state.body_tracks.keys()):
            body = state.body_tracks[bid]
            session = state.session_data.get(bid)

            if not session:
                continue

            cx, cy = body.get("center", (None, None))
            prev = body.get("prev_center")

            now = time.time()

            PERSISTENCE_WINDOW = 8  # seconds

            if now - body["last_seen"] > PERSISTENCE_WINDOW:
                session["exit_time"] = now
                session["force_exit"] = True
            
            # ---- Execute exit ----
            if session.get("force_exit") and not session.get("exit_processed", False):
                session["exit_processed"] = True
                session["force_exit"] = False
                print(f"[EXIT] Body {bid} left frame")

                session["last_body_box"] = body["box"]

                with state.lock:
                    state.lost_tracks[bid] = body
                    state.body_tracks.pop(bid, None)

        for bid, session in list(state.session_data.items()):
            if session.get("finalized") and time.time() > session.get("cooldown_until", 0):
                state.session_data.pop(bid, None)

            if bid in state.body_tracks:
                continue  # body still present

            # wait for recognition OR exit timeout
            if not session.get("identity_locked"):

                if session.get("exit_time") is None:
                    continue

                # allow recognition stabilization window
                if time.time() - session.get("exit_time", 0) < 2:
                    continue

            # 🔥 Prevent instant exit classification
            MIN_SESSION_TIME = 3  # seconds

            if time.time() - session.get("start_time", 0) < MIN_SESSION_TIME:
                continue

            if not session.get("finalized", False):
                # prevent rapid re-finalization
                if time.time() < session.get("cooldown_until", 0):
                    continue
                session["finalized"] = True
                print(f"[FINALIZE] Body {bid} session closing")

                if session.get("snapshot_saved"):
                    continue

                try:
                    # ---------- CLASSIFICATION ----------

                    identity = session.get("identity")

                    # 1️⃣ AUTHORIZED
                    if session.get("authorized_confirmed", False):
                        label = identity
                        folder = AUTHORIZED_DIR
                        color = (0,255,0)

                    # 2️⃣ FACE WAS CLEAR BUT NO MATCH → UNKNOWN
                    elif session.get("clear", 0) > 2:
                        label = "UNKNOWN"
                        folder = UNAUTHORIZED_DIR
                        color = (0,0,255)

                    # 3️⃣ FACE NEVER CLEAR → UNIDENTIFIED
                    else:
                        label = "UNIDENTIFIED"
                        folder = OBSTRUCTED_DIR
                        color = (0,165,255)
                    
                    snap_frame = session.get("best_frame")

                    if snap_frame is None:
                        snap_frame = session.get("last_frame")

                    if snap_frame is None:
                        snap_frame = frame.copy()

                    # ---------- SAFE BOX SELECTION ----------
                    box = session.get("best_face_box")

                    if box is None:
                        box = session.get("last_face_box")

                    if box is None:
                        box = session.get("last_body_box")

                    if box is None:
                        print("[WARN] No box available → snapshot skipped")
                    else:
                        # ensure valid list of 4 ints
                        if isinstance(box, np.ndarray):
                            box = box.tolist()

                        if len(box) != 4:
                            print("[WARN] Invalid box → snapshot skipped")
                        else:
                            box = [int(x) for x in box]
                            print(f"[SNAPSHOT] {label} saved from Body {bid}")
                            save_snapshot(snap_frame, box, label, folder, color)
                            session["snapshot_saved"] = True

                except Exception as e:
                    print("Snapshot error:", e)

                finally:
                    with state.lock:
                        # keep session for possible reconnect
                        session["cooldown_until"] = time.time() + 8

                        if time.time() > session["cooldown_until"]:
                            state.session_data.pop(bid, None)

# -------------------------
# STREAM
# -------------------------
def generate_frames():
    TARGET_FPS = 24
    FRAME_TIME = 1.0 / TARGET_FPS

    while True:
        start = time.time()

        with state.lock:
            frame = None if state.display_frame is None else state.display_frame.copy()
            bodies = state.body_tracks.copy()
            sessions = state.session_data.copy()

        if frame is None:
            time.sleep(0.01)
            continue

        # 🟩 Draw tracking boxes
        for bid, body in bodies.items():
            
            # skip stale tracks
            if time.time() - body.get("last_seen", 0) > 1.2:
                continue

            session = sessions.get(bid)
            if not session:
                continue

            box = body["box"]
            label = session.get("identity")

            if not label or label == "Unknown":
                label = "DETECTING"

            # 🎨 color logic
            if label == "DETECTING":
                color = (0, 255, 255)
            elif label == "UNKNOWN":
                color = (0, 0, 255)
            else:
                color = (0, 255, 0)

            x1, y1, x2, y2 = map(int, box)

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

            cv2.putText(
                frame,
                label,
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_DUPLEX,
                0.6,
                color,
                2
            )

        # resize for browser
        # frame = cv2.resize(frame, (640, 380))

        ret, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
        if not ret:
            continue

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' +
               buffer.tobytes() +
               b'\r\n')

        elapsed = time.time() - start
        sleep_time = FRAME_TIME - elapsed
        if sleep_time > 0:
            time.sleep(sleep_time)

@app.route('/')
def index():
    return render_template_string("""
    <html><body style="background:#111;color:#eee;text-align:center">
    <h2>LIVE SECURITY FEED</h2>
    <img src="/video_feed" width="80%">
    </body></html>
    """)

@app.route('/video_feed')
def video_feed():
    return Response(
        generate_frames(),
        mimetype='multipart/x-mixed-replace; boundary=frame',
        headers={
            "Cache-Control": "no-cache, no-store, must-revalidate",
            "Pragma": "no-cache",
            "Expires": "0"
        }
    )

if __name__=="__main__":
    threading.Thread(target=camera_thread,daemon=True).start()
    threading.Thread(target=detection_thread,daemon=True).start()
    threading.Thread(target=recognition_worker,daemon=True).start()
    app.run(host='0.0.0.0',port=8000,threaded=True)