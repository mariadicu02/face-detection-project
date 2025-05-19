import os
import cv2
import dlib
import numpy as np
import warnings
import json
import requests
from glob import glob
from datetime import datetime

# =================== CONFIGURARE ===================
MOODLE_URL = "http://192.168.1.135/moodle/webservice/rest/server.php"
MOODLE_TOKEN = "5a987e634310317873b8831b2faeaded"
COURSE_ID = 2
STUDENT_DATA_FILE = "studenti.json"

# =================== SUPRIMARE WARNINGS ===================
os.environ["OPENCV_VIDEOIO_DEBUG"] = "0"
os.environ["GST_DEBUG"] = "0"
warnings.filterwarnings("ignore")

print("[INFO] Începem inițializarea...")

# =================== VERIFICĂ FISIERE NECESARE ===================
if not os.path.exists("shape_predictor_68_face_landmarks.dat") or not os.path.exists("dlib_face_recognition_resnet_model_v1.dat"):
    print("[EROARE] Fisierele .dat lipsesc!")
    exit()

if not os.path.exists(STUDENT_DATA_FILE):
    print("[EROARE] students.json lipsește!")
    exit()

# =================== ÎNCARCĂ DATE STUDENȚI ===================
with open(STUDENT_DATA_FILE, "r", encoding="utf-8") as f:
    student_data = json.load(f)

# =================== FUNCȚII MOODLE ===================
def call_moodle_function(function, params):
    url = f"{MOODLE_URL}?wstoken={MOODLE_TOKEN}&moodlewsrestformat=json&wsfunction={function}"
    response = requests.post(url, data=params)
    
    try:
        return response.json()
    except Exception as e:
        print("[EROARE JSON] Răspuns invalid de la server!")
        print("[HTTP STATUS]:", response.status_code)
        print("[RESPONSE TEXT]:", response.text[:500])  # doar primele 500 caractere
        raise

def get_attendance_id(course_id):
    contents = call_moodle_function("core_course_get_contents", {"courseid": course_id})
    for section in contents:
        for module in section.get("modules", []):
            if module["modname"] == "attendance":
                return module["instance"]
    return None

def get_latest_session_id(attendance_id):
    sessions = call_moodle_function("mod_attendance_get_sessions", {"attendanceid": attendance_id})
    now_timestamp = datetime.now().timestamp()
    for s in sessions.get("sessions", []):
        if s["sessdate"] <= now_timestamp <= (s["sessdate"] + s["duration"]):
            return s["id"]
    return None

def get_present_status_id(attendance_id):
    statuses = call_moodle_function("mod_attendance_get_user_statuses", {"attendanceid": attendance_id})
    for status in statuses.get("statuses", []):
        if status["acronym"].lower() == "p":  # P = Prezent
            return status["id"]
    return None

def mark_attendance(user_id, session_id, status_id):
    params = {
        "sessionid": session_id,
        "userids[0]": user_id,
        "statusid": status_id
    }
    response = call_moodle_function("mod_attendance_update_user_status", params)
    return response

# =================== INIȚIALIZARE RECUNOAȘTERE FACIALĂ ===================
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
recognizer = dlib.face_recognition_model_v1("dlib_face_recognition_resnet_model_v1.dat")

faces = []
label_names = {}

print("[INFO] Încărcăm imaginile din 'poze/'...")

for idx, filename in enumerate(glob("poze/*.*")):
    img = cv2.imread(filename)
    if img is None:
        continue

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    detected_faces = detector(gray)
    if len(detected_faces) == 0:
        continue

    shape = predictor(img, detected_faces[0])
    descriptor = recognizer.compute_face_descriptor(img, shape)

    name = os.path.splitext(os.path.basename(filename))[0]
    if name in student_data:
        faces.append(descriptor)
        label_names[idx] = name
        print(f"[INFO] Imagine {name} încărcată.")
    else:
        print(f"[AVERTISMENT] {name} nu există în students.json")

# =================== PORNEȘTE CAMERA ===================
print("[INFO] Pornim camera...")

cap = cv2.VideoCapture("/dev/video0", cv2.CAP_V4L2)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)
cap.set(cv2.CAP_PROP_FPS, 10)

if not cap.isOpened():
    print("[EROARE] Nu se poate accesa camera.")
    exit()

frame_count = 0
process_every_n_frames = 5
studenti_marcati = set()

# =================== BUCLE VIDEO PRINCIPAL ===================
while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    if frame_count % process_every_n_frames == 0:
        detected_faces = detector(gray)

        for face in detected_faces:
            shape = predictor(frame, face)
            descriptor = recognizer.compute_face_descriptor(frame, shape)

            distances = [np.linalg.norm(np.array(descriptor) - np.array(f)) for f in faces]
            min_dist = min(distances) if distances else float("inf")
            id_ = distances.index(min_dist) if distances else -1

            name = label_names.get(id_, "Necunoscut") if min_dist < 0.6 else "Necunoscut"
            color = (0, 255, 0) if name != "Necunoscut" else (0, 0, 255)

            cv2.rectangle(frame, (face.left(), face.top()), (face.right(), face.bottom()), color, 2)
            cv2.putText(frame, name, (face.left(), face.top() - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            if name != "Necunoscut" and name not in studenti_marcati:
                student_id = int(student_data[name]["id"])
                print(f"[RECUNOSCUT ✅] {name} (ID: {student_id}) a fost identificat.")

                # === MOODLE ===
                attendance_id = get_attendance_id(COURSE_ID)
                if attendance_id:
                    session_id = get_latest_session_id(attendance_id)
                    status_id = get_present_status_id(attendance_id)
                    if session_id and status_id:
                        result = mark_attendance(student_id, session_id, status_id)
                        print(f"[MOODLE ✅] Prezență marcată pentru {name}.")
                    else:
                        print(f"[MOODLE ⚠️] Nu s-a găsit sesiunea activă sau statusul.")
                else:
                    print("[MOODLE ❌] Nu s-a găsit activitatea de prezență.")
                
                studenti_marcati.add(name)

    frame_count += 1
    cv2.imshow('Live - Recunoastere Faciala', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# =================== CURĂȚARE ===================
cap.release()
cv2.destroyAllWindows()
cv2.waitKey(1)
