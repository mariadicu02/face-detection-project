import os
import cv2
import dlib
import numpy as np
import warnings
import json
import requests
import threading
import logging
from glob import glob
from datetime import datetime

# =================== CONFIG ===================
MOODLE_URL = "http://192.168.0.101/moodle/webservice/rest/server.php"
MOODLE_TOKEN = os.getenv("MOODLE_TOKEN", "5a987e634310317873b8831b2faeaded")
COURSE_ID = 2
STUDENT_DATA_FILE = "studenti.json"
FACE_MATCH_THRESHOLD = 0.6
LOCAL_BACKUP_FILE = "prezenta_backup.json"

# =================== LOGGING ===================
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

# =================== SUPPRESS WARNINGS ===================
os.environ["OPENCV_VIDEOIO_DEBUG"] = "0"
os.environ["GST_DEBUG"] = "0"
warnings.filterwarnings("ignore")

logging.info("Initializare sistem...")

# =================== VERIFY FILES ===================
REQUIRED_FILES = [
    "shape_predictor_68_face_landmarks.dat",
    "dlib_face_recognition_resnet_model_v1.dat",
    STUDENT_DATA_FILE
]

for file in REQUIRED_FILES:
    if not os.path.exists(file):
        logging.error(f"Fișier lipsă: {file}")
        exit()

# =================== NFC FUNCTION ===================
def read_nfc_uid():
    try:
        import board
        import busio
        import adafruit_pn532.i2c

        i2c = busio.I2C(board.SCL, board.SDA)
        pn532 = adafruit_pn532.i2c.PN532_I2C(i2c, debug=False)
        pn532.SAM_configuration()

        logging.info("Aștept card NFC...")
        uid = pn532.read_passive_target(timeout=2)
        if uid:
            logging.info(f"Card detectat: {uid.hex().upper()}")
            return uid.hex().upper()
        else:
            logging.warning("Card NFC nedetectat.")
            return None
    except Exception as e:
        logging.error(f"Eroare la citire NFC: {e}")
        return None

# =================== LOAD STUDENT DATA ===================
with open(STUDENT_DATA_FILE, "r", encoding="utf-8") as f:
    student_data = json.load(f)

# =================== MOODLE API ===================
def call_moodle_function(function, params):
    url = f"{MOODLE_URL}?wstoken={MOODLE_TOKEN}&moodlewsrestformat=json&wsfunction={function}"
    try:
        response = requests.post(url, data=params, timeout=5)
        data = response.json()
        if "exception" in data:
            logging.error(f"Eroare Moodle: {data.get('message')}")
        return data
    except Exception as e:
        logging.error(f"Conexiune Moodle eșuată: {e}")
        return {}

def get_attendance_id(course_id):
    contents = call_moodle_function("core_course_get_contents", {"courseid": course_id})
    for section in contents:
        for module in section.get("modules", []):
            if module["modname"] == "attendance":
                return module["instance"]
    return None

def get_latest_session_id(attendance_id):
    sessions_response = call_moodle_function("mod_attendance_get_sessions", {"attendanceid": attendance_id})
    sessions = sessions_response.get("sessions", []) if isinstance(sessions_response, dict) else sessions_response

    now = datetime.now().timestamp()
    for session in sessions:
        if session["sessdate"] <= now <= (session["sessdate"] + session["duration"]):
            return session["id"]
    return None

def get_present_status_id_from_session(session_id):
    response = call_moodle_function("mod_attendance_get_session", {"sessionid": session_id})
    for status in response.get("statuses", []):
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

def backup_local_attendance(name, student_id):
    data = {
        "name": name,
        "id": student_id,
        "timestamp": datetime.now().isoformat()
    }

    if os.path.exists(LOCAL_BACKUP_FILE):
        with open(LOCAL_BACKUP_FILE, "r", encoding="utf-8") as f:
            backup = json.load(f)
    else:
        backup = {"backup_attendance": []}

    backup["backup_attendance"].append(data)

    with open(LOCAL_BACKUP_FILE, "w", encoding="utf-8") as f:
        json.dump(backup, f, indent=2, ensure_ascii=False)

    logging.info(f"Prezență salvată local pentru {name}.")

def mark_student_attendance(name, student_id):
    attendance_id = get_attendance_id(COURSE_ID)
    session_id = get_latest_session_id(attendance_id)
    status_id = get_present_status_id_from_session(session_id)

    logging.debug(f"Attendance ID: {attendance_id}")
    logging.debug(f"Session ID: {session_id}")
    logging.debug(f"Status ID: {status_id}")

    if attendance_id and session_id and status_id:
        result = mark_attendance(student_id, session_id, status_id)
        if "status" in result or result == {}:
            logging.info(f"[MOODLE ✅] {name} marcat prezent.")
        else:
            logging.warning(f"[MOODLE ⚠️] Eroare la marcare. Salvare locală.")
            backup_local_attendance(name, student_id)
    else:
        logging.warning("[MOODLE ❌] Informații lipsă. Salvare locală.")
        backup_local_attendance(name, student_id)

# =================== LOAD FACE DATA ===================
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
recognizer = dlib.face_recognition_model_v1("dlib_face_recognition_resnet_model_v1.dat")

faces = []
label_names = {}
logging.info("Încarc imagini din folderul 'poze/'...")

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
        logging.info(f"Încărcat: {name}")
    else:
        logging.warning(f"{name} nu există în fișierul {STUDENT_DATA_FILE}.")

# =================== CAMERA LOOP ===================
cap = cv2.VideoCapture("/dev/video0")
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)
cap.set(cv2.CAP_PROP_FPS, 10)

if not cap.isOpened():
    logging.error("Nu pot accesa camera.")
    exit()

frame_count = 0
studenti_marcati = set()

logging.info("Camera pornită.")

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
            name = label_names.get(idx, "Necunoscut") if min_dist < FACE_MATCH_THRESHOLD else "Necunoscut"
            color = (0, 255, 0) if name != "Necunoscut" else (0, 0, 255)

            cv2.rectangle(frame, (face.left(), face.top()), (face.right(), face.bottom()), color, 2)
            cv2.putText(frame, name, (face.left(), face.top() - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            if name != "Necunoscut" and name not in studenti_marcati:
                student_id = int(student_data[name]["id"])
                threading.Thread(target=mark_student_attendance, args=(name, student_id)).start()
                studenti_marcati.add(name)

            if name == "Necunoscut":
                logging.info("Persoană necunoscută. Se cere card NFC.")
                uid = read_nfc_uid()
                if uid:
                    for key, info in student_data.items():
                        if info.get("nfc_uid", "").upper() == uid.upper():
                            if key not in studenti_marcati:
                                student_id = int(info["id"])
                                logging.info(f"NFC recunoscut: {key} (ID: {student_id})")
                                threading.Thread(target=mark_student_attendance, args=(key, student_id)).start()
                                studenti_marcati.add(key)
                            break
                    else:
                        logging.warning("Card NFC necunoscut.")

    frame_count += 1
    cv2.imshow("Live - Facial Recognition", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
cv2.waitKey(1)
