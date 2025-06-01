# facial_attendance_moodle.py

import os
import cv2
import dlib
import numpy as np
import warnings
import json
import requests
import threading
from glob import glob
from datetime import datetime

# =================== CONFIG ===================
MOODLE_URL = "http://192.168.0.101/moodle/webservice/rest/server.php"
MOODLE_TOKEN = os.getenv("MOODLE_TOKEN", "5a987e634310317873b8831b2faeaded")
COURSE_ID = 2
STUDENT_DATA_FILE = "studenti.json"

# =================== SUPPRESS WARNINGS ===================
os.environ["OPENCV_VIDEOIO_DEBUG"] = "0"
os.environ["GST_DEBUG"] = "0"
warnings.filterwarnings("ignore")

print("[INFO] Initializing...")

# =================== VERIFY FILES ===================
REQUIRED_FILES = [
    "shape_predictor_68_face_landmarks.dat",
    "dlib_face_recognition_resnet_model_v1.dat",
    STUDENT_DATA_FILE
]

for file in REQUIRED_FILES:
    if not os.path.exists(file):
        print(f"[ERROR] Required file missing: {file}")
        exit()

# =================== LOAD STUDENT DATA ===================
with open(STUDENT_DATA_FILE, "r", encoding="utf-8") as f:
    student_data = json.load(f)

# =================== MOODLE API CALL ===================
def call_moodle_function(function, params):
    url = f"{MOODLE_URL}?wstoken={MOODLE_TOKEN}&moodlewsrestformat=json&wsfunction={function}"
    try:
        response = requests.post(url, data=params, timeout=5)
        print(f"[DEBUG] URL: {url}")
        print(f"[DEBUG] Params: {params}")
        print(f"[DEBUG] Status Code: {response.status_code}")
        return response.json()
    except Exception as e:
        print("[ERROR] Moodle request failed:", e)
        return {}

# =================== MOODLE HELPERS ===================
def get_attendance_id(course_id):
    contents = call_moodle_function("core_course_get_contents", {"courseid": course_id})
    for section in contents:
        for module in section.get("modules", []):
            if module["modname"] == "attendance":
                return module["instance"]
    return None

def get_latest_session_id(attendance_id):
    sessions_response = call_moodle_function("mod_attendance_get_sessions", {"attendanceid": attendance_id})

    if isinstance(sessions_response, list):
        sessions = sessions_response
    else:
        sessions = sessions_response.get("sessions", [])

    now = datetime.now().timestamp()
    for session in sessions:
        if session["sessdate"] <= now <= (session["sessdate"] + session["duration"]):
            return session["id"]
    return None


def get_present_status_id_from_session(session_id):
    response = call_moodle_function("mod_attendance_get_session", {"sessionid": session_id})
    statuses = response.get("statuses", [])
    for status in statuses:
        if status["acronym"].lower() == "p":
            return status["id"]
    return None

def mark_attendance(user_id, session_id, status_id):
    params = {
        'sessionid': session_id,
        'updates[0][studentid]': user_id,
        'updates[0][statusid]': status_id
    }

    return call_moodle_function("mod_attendance_update_user_status", params)



'''def mark_student_attendance(name, student_id):
    attendance_id = get_attendance_id(COURSE_ID)
    print("[DEBUG] Attendance ID:", attendance_id)
    if attendance_id:
        session_id = get_latest_session_id(attendance_id)
        print("[DEBUG] Session ID:", session_id)
        status_id = get_present_status_id(attendance_id)
        print("[DEBUG] Status ID:", status_id)

        if session_id and status_id:
            result = mark_attendance(student_id, session_id, status_id)
            print(f"[MOODLE ✅] Marked {name} present. Response: {result}")
        else:
            print(f"[MOODLE ⚠️] Session or status not found.")
    else:
        print("[MOODLE ❌] Attendance activity not found.")
'''
def mark_student_attendance(name, student_id):
    attendance_id = get_attendance_id(COURSE_ID)
    session_id = get_latest_session_id(attendance_id)
    print("[DEBUG] Attendance ID:", attendance_id)
    print("[DEBUG] Session ID:", session_id)

    status_id = get_present_status_id_from_session(session_id)
    print("[DEBUG] Status ID:", status_id)

    if status_id:
        result = mark_attendance(student_id, session_id, status_id)
        print(f"[MOODLE ✅] Marked {name} present. Response: {result}")
    else:
        print(f"[MOODLE ⚠️] Status not found.")

# =================== LOAD FACE DATA ===================
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
recognizer = dlib.face_recognition_model_v1("dlib_face_recognition_resnet_model_v1.dat")

faces = []
label_names = {}
print("[INFO] Loading images from 'poze/'...")

for idx, filename in enumerate(glob("poze/*.*")):
    img = cv2.imread(filename)
    if img is None:
        continue

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    detected = detector(gray)
    if not detected:
        continue

    shape = predictor(img, detected[0])
    descriptor = recognizer.compute_face_descriptor(img, shape)

    name = os.path.splitext(os.path.basename(filename))[0]
    if name in student_data:
        faces.append(descriptor)
        label_names[idx] = name
        print(f"[INFO] Loaded {name}.")
    else:
        print(f"[WARNING] {name} not found in {STUDENT_DATA_FILE}.")

# =================== START CAMERA ===================
cap = cv2.VideoCapture("/dev/video0")
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)
cap.set(cv2.CAP_PROP_FPS, 10)

if not cap.isOpened():
    print("[ERROR] Cannot access camera.")
    exit()

frame_count = 0
studenti_marcati = set()

print("[INFO] Camera started.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    if frame_count % 5 == 0:
        detected_faces = detector(gray)
        for face in detected_faces:
            shape = predictor(frame, face)
            descriptor = recognizer.compute_face_descriptor(frame, shape)

            distances = [np.linalg.norm(np.array(descriptor) - np.array(f)) for f in faces]
            min_dist = min(distances) if distances else float("inf")
            idx = distances.index(min_dist) if distances else -1
            name = label_names.get(idx, "Necunoscut") if min_dist < 0.6 else "Necunoscut"
            color = (0, 255, 0) if name != "Necunoscut" else (0, 0, 255)

            cv2.rectangle(frame, (face.left(), face.top()), (face.right(), face.bottom()), color, 2)
            cv2.putText(frame, name, (face.left(), face.top() - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            if name != "Necunoscut" and name not in studenti_marcati:
                student_id = int(student_data[name]["id"])
                print(f"[RECOGNIZED ✅] {name} (ID: {student_id})")
                threading.Thread(target=mark_student_attendance, args=(name, student_id)).start()
                studenti_marcati.add(name)

    frame_count += 1
    cv2.imshow("Live - Facial Recognition", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
cv2.waitKey(1)
