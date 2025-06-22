import os
import cv2
import dlib
import numpy as np
import warnings
import sqlite3
import json
import requests
import threading
import logging
from glob import glob
from datetime import datetime
import urllib.request
from pathlib import Path
from RPLCD.i2c import CharLCD
from time import sleep
from dotenv import load_dotenv


# =================== LCD Setup ===================
lcd = CharLCD(i2c_expander='PCF8574', address=0x27, port=1, cols=16, rows=2, charmap='A00', auto_linebreaks=True)

def afiseaza_mesaj(text, durata=3):
    lcd.clear()
    lcd.write_string(text)
    sleep(durata)
    lcd.clear()
    
    
load_dotenv()

# =================== CONFIG ===================
MOODLE_URL = os.getenv("MOODLE_URL")
MOODLE_TOKEN = os.getenv("MOODLE_TOKEN")
COURSE_ID = int(os.getenv("COURSE_ID", 2))
FACE_MATCH_THRESHOLD = float(os.getenv("FACE_MATCH_THRESHOLD", 0.6))
PHOTOS_DIR = os.getenv("PHOTOS_DIR", "poze_moodle")



LOCAL_DB_FILE = "prezenta.db"
# =================== LOGGING ===================
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

# =================== SUPPRESS WARNINGS ===================
os.environ["OPENCV_VIDEOIO_DEBUG"] = "0"
os.environ["GST_DEBUG"] = "0"
warnings.filterwarnings("ignore")

logging.info("Initializare sistem prezență...")

# =================== DATABASE SETUP ===================
def init_database():
    """Inițializează baza de date SQLite cu tabelele necesare"""
    conn = sqlite3.connect(LOCAL_DB_FILE)
    c = conn.cursor()
    
    # Tabel pentru studenți (sincronizat din Moodle)
    c.execute('''CREATE TABLE IF NOT EXISTS studenti (
        id INTEGER PRIMARY KEY,
        nume TEXT NOT NULL,
        prenume TEXT NOT NULL,
        email TEXT,
        nfc_uid TEXT,
        poza_path TEXT,
        last_sync TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )''')
    
    # Tabel pentru sesiuni de prezență
    c.execute('''CREATE TABLE IF NOT EXISTS sesiuni (
        id INTEGER PRIMARY KEY,
        moodle_session_id INTEGER,
        nume_sesiune TEXT,
        data_start TIMESTAMP,
        data_end TIMESTAMP,
        curs_id INTEGER,
        activa INTEGER DEFAULT 0
    )''')
    
    # Tabel pentru prezențe
    c.execute('''CREATE TABLE IF NOT EXISTS prezente (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        student_id INTEGER,
        sesiune_id INTEGER,
        metoda_detectie TEXT,  -- 'camera' sau 'nfc'
        timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (student_id) REFERENCES studenti (id),
        FOREIGN KEY (sesiune_id) REFERENCES sesiuni (id)
    )''')
    
    conn.commit()
    conn.close()
    logging.info("Baza de date inițializată cu succes.")

# =================== MOODLE API FUNCTIONS ===================
def call_moodle_function(function, params):
    """Apelează o funcție din API-ul Moodle"""
    url = f"{MOODLE_URL}?wstoken={MOODLE_TOKEN}&moodlewsrestformat=json&wsfunction={function}"
    try:
        response = requests.post(url, data=params, timeout=10)
        data = response.json()
        if "exception" in data:
            logging.error(f"Eroare Moodle: {data.get('message')}")
        return data
    except Exception as e:
        logging.error(f"Conexiune Moodle eșuată: {e}")
        return {}

def get_enrolled_students(course_id):
    """Extrage lista studenților înrolați în curs"""
    logging.info("Extrag studenții înrolați din Moodle...")
    params = {'courseid': course_id}
    students = call_moodle_function("core_enrol_get_enrolled_users", params)
    
    if not isinstance(students, list):
        logging.error("Nu s-au putut extrage studenții din Moodle")
        return []
    
    student_list = []
    for student in students:
        # Filtrăm doar studenții (nu profesorii)
        if any(role['shortname'] == 'student' for role in student.get('roles', [])):
            student_info = {
                'id': student['id'],
                'nume': student['lastname'],
                'prenume': student['firstname'],
                'email': student['email'],
                'profile_image_url': student.get('profileimageurl', '')
            }
            student_list.append(student_info)
    
    logging.info(f"Găsiți {len(student_list)} studenți înrolați.")
    return student_list

def download_profile_image(url, student_id):
    """Descarcă poza de profil a unui student"""
    if not url or 'gravatar' in url:  # Skip default avatars
        return None
    
    try:
        os.makedirs(PHOTOS_DIR, exist_ok=True)
        filename = f"{PHOTOS_DIR}/student_{student_id}.jpg"
        
        # Adaugă token-ul Moodle la URL dacă e necesar
        if MOODLE_TOKEN and 'token=' not in url:
            separator = '&' if '?' in url else '?'
            url = f"{url}{separator}token={MOODLE_TOKEN}"
        
        urllib.request.urlretrieve(url, filename)
        logging.info(f"Poza descărcată pentru studentul ID {student_id}")
        return filename
    except Exception as e:
        logging.warning(f"Nu s-a putut descărca poza pentru studentul {student_id}: {e}")
        return None

def sync_students_from_moodle():
    """Sincronizează studenții din Moodle în baza de date locală"""
    students = get_enrolled_students(COURSE_ID)
    if not students:
        return
    
    conn = sqlite3.connect(LOCAL_DB_FILE)
    c = conn.cursor()
    
    for student in students:
        # Descarcă poza de profil
        photo_path = None
        if student['profile_image_url']:
            photo_path = download_profile_image(student['profile_image_url'], student['id'])
        
        # Inserează sau actualizează studentul
        # Păstrează UID-ul existent
        c.execute("SELECT nfc_uid FROM studenti WHERE id = ?", (student['id'],))
        existing = c.fetchone()
        nfc_uid = existing[0] if existing else None

        c.execute('''INSERT OR REPLACE INTO studenti 
                     (id, nume, prenume, email, nfc_uid, poza_path, last_sync) 
                     VALUES (?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)''',
                  (student['id'], student['nume'], student['prenume'], 
                   student['email'], nfc_uid, photo_path))

    
    conn.commit()
    conn.close()
    logging.info("Sincronizare studenți completă.")

def get_attendance_sessions():
    """Extrage sesiunile de prezență active din Moodle"""
    # Găsește modulul de prezență
    contents = call_moodle_function("core_course_get_contents", {"courseid": COURSE_ID})
    attendance_id = None
    
    for section in contents:
        for module in section.get("modules", []):
            if module["modname"] == "attendance":
                attendance_id = module["instance"]
                break
    
    if not attendance_id:
        logging.warning("Nu s-a găsit modul de prezență în curs.")
        return []
    
    # Extrage sesiunile
    sessions_response = call_moodle_function("mod_attendance_get_sessions", 
                                           {"attendanceid": attendance_id})
    sessions = sessions_response.get("sessions", []) if isinstance(sessions_response, dict) else sessions_response
    
    return sessions

def sync_attendance_sessions():
    """Sincronizează sesiunile de prezență din Moodle"""
    sessions = get_attendance_sessions()
    if not sessions:
        return
    
    conn = sqlite3.connect(LOCAL_DB_FILE)
    c = conn.cursor()
    
    now = datetime.now().timestamp()
    
    for session in sessions:
        start_time = datetime.fromtimestamp(session["sessdate"])
        end_time = datetime.fromtimestamp(session["sessdate"] + session["duration"])
        
        # Verifică dacă sesiunea este activă
        is_active = session["sessdate"] <= now <= (session["sessdate"] + session["duration"])
        
        c.execute('''INSERT OR REPLACE INTO sesiuni 
                     (moodle_session_id, nume_sesiune, data_start, data_end, curs_id, activa)
                     VALUES (?, ?, ?, ?, ?, ?)''',
                  (session["id"], session.get("description", f"Sesiune {session['id']}"),
                   start_time, end_time, COURSE_ID, 1 if is_active else 0))
    
    conn.commit()
    conn.close()
    logging.info("Sincronizare sesiuni completă.")

def get_active_session():
    """Returnează sesiunea activă curentă"""
    conn = sqlite3.connect(LOCAL_DB_FILE)
    c = conn.cursor()
    c.execute("SELECT * FROM sesiuni WHERE activa = 1 LIMIT 1")
    session = c.fetchone()
    conn.close()
    return session

def mark_attendance_local(student_id, method="camera"):
    """Marchează prezența în baza de date locală"""
    active_session = get_active_session()
    if not active_session:
        logging.warning("Nu există sesiune activă pentru a marca prezența.")
        return False
    
    conn = sqlite3.connect(LOCAL_DB_FILE)
    c = conn.cursor()
    
    # Verifică dacă studentul a fost deja marcat în această sesiune
    c.execute('''SELECT id FROM prezente 
                 WHERE student_id = ? AND sesiune_id = ?''',
              (student_id, active_session[0]))
    
    if c.fetchone():
        logging.info(f"Studentul ID {student_id} a fost deja marcat prezent în această sesiune.")
        conn.close()
        return False
    
    # Marchează prezența
    c.execute('''INSERT INTO prezente (student_id, sesiune_id, metoda_detectie)
                 VALUES (?, ?, ?)''',
              (student_id, active_session[0], method))
    
    conn.commit()
    conn.close()
    
    # Obține numele studentului pentru log
    conn = sqlite3.connect(LOCAL_DB_FILE)
    c = conn.cursor()
    c.execute("SELECT nume, prenume FROM studenti WHERE id = ?", (student_id,))
    student_info = c.fetchone()
    conn.close()
    
    if student_info:
        logging.info(f"[PREZENȚĂ ✅] {student_info[1]} {student_info[0]} marcat prezent prin {method}")
    
    return True

# =================== NFC FUNCTION ===================
def read_nfc_uid():
    """Citește UID-ul unui card NFC"""
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
            uid_str = uid.hex().upper()
            logging.info(f"Card detectat: {uid_str}")
            return uid_str
        else:
            logging.warning("Card NFC nedetectat.")
            return None
    except Exception as e:
        logging.error(f"Eroare la citire NFC: {e}")
        return None

def find_student_by_nfc(uid):
    """Găsește un student după UID-ul NFC"""
    conn = sqlite3.connect(LOCAL_DB_FILE)
    c = conn.cursor()
    c.execute("SELECT id, nume, prenume FROM studenti WHERE nfc_uid = ?", (uid,))
    student = c.fetchone()
    conn.close()
    return student

# =================== CAMERA INITIALIZATION ===================
def initialize_camera():
    """Inițializează camera cu mai multe încercări și backend-uri diferite"""
    logging.info("Încerc să inițializez camera...")
    
    # Lista de backend-uri și indici de cameră de încercat
    backends = [cv2.CAP_V4L2, cv2.CAP_GSTREAMER, cv2.CAP_ANY]
    camera_indices = [0, 1, -1, '/dev/video0', '/dev/video1']
    
    for backend in backends:
        backend_name = {cv2.CAP_V4L2: "V4L2", cv2.CAP_GSTREAMER: "GStreamer", cv2.CAP_ANY: "ANY"}.get(backend, str(backend))
        logging.info(f"Încerc backend-ul: {backend_name}")
        
        for camera_index in camera_indices:
            try:
                logging.info(f"  Încerc camera index: {camera_index}")
                
                # Pentru stringuri (path-uri de device), folosim doar CAP_V4L2
                if isinstance(camera_index, str) and backend != cv2.CAP_V4L2:
                    continue
                
                cap = cv2.VideoCapture(camera_index, backend)
                
                if cap.isOpened():
                    # Testează dacă putem citi un frame
                    ret, frame = cap.read()
                    if ret and frame is not None:
                        logging.info(f"✅ Camera inițializată cu succes: index {camera_index}, backend {backend_name}")
                        
                        # Setează proprietăți optime
                        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                        cap.set(cv2.CAP_PROP_FPS, 30)
                        
                        # Verifică proprietățile setate
                        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        fps = int(cap.get(cv2.CAP_PROP_FPS))
                        
                        logging.info(f"Rezoluție cameră: {width}x{height}, FPS: {fps}")
                        return cap
                    else:
                        logging.warning(f"  Camera {camera_index} s-a deschis dar nu poate citi frame-uri")
                        cap.release()
                else:
                    logging.warning(f"  Nu pot deschide camera {camera_index}")
                    
            except Exception as e:
                logging.warning(f"  Eroare la inițializarea camerei {camera_index}: {e}")
                continue
    


# =================== FACE RECOGNITION SETUP ===================
def load_face_data():
    """Încarcă datele pentru recunoașterea facială din pozele studenților"""
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
    recognizer = dlib.face_recognition_model_v1("dlib_face_recognition_resnet_model_v1.dat")
    
    faces = []
    student_ids = []
    
    conn = sqlite3.connect(LOCAL_DB_FILE)
    c = conn.cursor()
    c.execute("SELECT id, nume, prenume, poza_path FROM studenti WHERE poza_path IS NOT NULL")
    students_with_photos = c.fetchall()
    conn.close()
    
    logging.info("Încarc datele pentru recunoașterea facială...")
    
    for student_id, nume, prenume, photo_path in students_with_photos:
        if not os.path.exists(photo_path):
            continue
            
        img = cv2.imread(photo_path)
        if img is None:
            continue

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        detected = detector(gray)
        if not detected:
            continue

        shape = predictor(gray, detected[0])
        descriptor = recognizer.compute_face_descriptor(img, shape)

        faces.append(descriptor)
        student_ids.append(student_id)
        logging.info(f"Încărcat: {prenume} {nume} (ID: {student_id})")
    
    return detector, predictor, recognizer, faces, student_ids

# =================== MAIN LOOP ===================
def main():
    # Inițializare
    init_database()
    
    # Sincronizare cu Moodle
    logging.info("Sincronizez datele cu Moodle...")
    sync_students_from_moodle()
    sync_attendance_sessions()
    
    active_session = get_active_session()
    if not active_session:
        logging.error("Nu există sesiune activă de prezență!")
        return
    
    logging.info(f"Sesiune activă: {active_session[2]} ({active_session[3]} - {active_session[4]})")
    afiseaza_mesaj("Sesiune activa!")
    
    # Încărcare date pentru recunoașterea facială
    try:
        detector, predictor, recognizer, faces, student_ids = load_face_data()
    except Exception as e:
        logging.error(f"Eroare la încărcarea datelor pentru recunoașterea facială: {e}")
        return
    
    
    # Inițializare cameră
    camera_result = initialize_camera()
    if camera_result is None:
        logging.error("Nu pot accesa camera. Sistemul se oprește.")
        return
    
    # Verifică dacă avem PiCamera sau OpenCV VideoCapture
    if isinstance(camera_result, tuple):
        # PiCamera
        camera, rawCapture = camera_result
        logging.info("Folosesc PiCamera pentru captura video")
        use_picamera = True
    else:
        # OpenCV VideoCapture
        cap = camera_result
        logging.info("Folosesc OpenCV VideoCapture pentru captura video")
        use_picamera = False

    frame_count = 0
    studenti_marcati = set()
    
    logging.info("Camera pornită. Sistem de prezență activ!")
    
    try:
        if use_picamera:
            # Loop pentru PiCamera
            from picamera.array import PiRGBArray
            for frame_data in camera.capture_continuous(rawCapture, format="bgr", use_video_port=True):
                frame = frame_data.array
                
                # Procesare frame
                frame = cv2.flip(frame, 1)
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

                if frame_count % 5 == 0:  # Procesează la fiecare 5 cadre
                    detected_faces = detector(gray)
                    for face in detected_faces:
                        shape = predictor(gray, face)
                        descriptor = recognizer.compute_face_descriptor(frame, shape)

                        # Compară cu fețele cunoscute
                        if faces:
                            distances = [np.linalg.norm(np.array(descriptor) - np.array(f)) for f in faces]
                            min_dist = min(distances)
                            best_match_idx = distances.index(min_dist)
                            
                            if min_dist < FACE_MATCH_THRESHOLD:
                                student_id = student_ids[best_match_idx]
                                
                                # Obține informații despre student
                                conn = sqlite3.connect(LOCAL_DB_FILE)
                                c = conn.cursor()
                                c.execute("SELECT nume, prenume FROM studenti WHERE id = ?", (student_id,))
                                student_info = c.fetchone()
                                conn.close()
                                
                                if student_info and student_id not in studenti_marcati:
                                    name = f"{student_info[1]} {student_info[0]}"
                                    threading.Thread(target=mark_attendance_local, 
                                                   args=(student_id, "camera")).start()
                                    studenti_marcati.add(student_id)
                                    
                                color = (0, 255, 0)  # Verde pentru recunoscut
                                label = f"{student_info[1]} {student_info[0]}" if student_info else f"ID: {student_id}"
                            else:
                                color = (0, 0, 255)  # Roșu pentru necunoscut
                                label = "Necunoscut"
                                
                                # Încearcă citirea NFC pentru persoanele necunoscute
                                logging.info("Persoană necunoscută detectată. Se încearcă citirea NFC...")
                                uid = read_nfc_uid()
                                if not uid:
                                    sleep(1)
                                    uid = read_nfc_uid()
                                    student = find_student_by_nfc(uid)
                                    if student and student[0] not in studenti_marcati:
                                        threading.Thread(target=mark_attendance_local, 
                                                       args=(student[0], "nfc")).start()
                                        studenti_marcati.add(student[0])
                                        nume_complet = f"{student[2]} {student[1]}"
                                        label = f"{nume_complet} (NFC)"
                                        color = (255, 255, 0)  # Galben pentru NFC
                                        afiseaza_mesaj(nume_complet)

                        else:
                            color = (0, 0, 255)
                            label = "Necunoscut"

                        # Desenează dreptunghiul și eticheta
                        cv2.rectangle(frame, (face.left(), face.top()), 
                                    (face.right(), face.bottom()), color, 2)
                        cv2.putText(frame, label, (face.left(), face.top() - 10), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

                frame_count += 1
                cv2.imshow("Sistem Prezență - Recunoaștere Facială", frame)
                
                # Curăță buffer-ul pentru următorul frame
                rawCapture.truncate(0)
                
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                    
        else:
            # Loop pentru OpenCV VideoCapture
            while True:
                ret, frame = cap.read()
                if not ret:
                    logging.error("Nu pot citi frame din cameră")
                    break

                frame = cv2.flip(frame, 1)
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

                if frame_count % 5 == 0:  # Procesează la fiecare 5 cadre
                    detected_faces = detector(gray)
                    for face in detected_faces:
                        shape = predictor(gray, face)
                        descriptor = recognizer.compute_face_descriptor(frame, shape)

                        # Compară cu fețele cunoscute
                        if faces:
                            distances = [np.linalg.norm(np.array(descriptor) - np.array(f)) for f in faces]
                            min_dist = min(distances)
                            best_match_idx = distances.index(min_dist)
                            
                            if min_dist < FACE_MATCH_THRESHOLD:
                                student_id = student_ids[best_match_idx]
                                
                                # Obține informații despre student
                                conn = sqlite3.connect(LOCAL_DB_FILE)
                                c = conn.cursor()
                                c.execute("SELECT nume, prenume FROM studenti WHERE id = ?", (student_id,))
                                student_info = c.fetchone()
                                conn.close()
                                
                                if student_info and student_id not in studenti_marcati:
                                    name = f"{student_info[1]} {student_info[0]}"
                                    threading.Thread(target=mark_attendance_local, 
                                                   args=(student_id, "camera")).start()
                                    studenti_marcati.add(student_id)
                                    afiseaza_mesaj(name)
                                    
                                color = (0, 255, 0)  # Verde pentru recunoscut
                                label = f"{student_info[1]} {student_info[0]}" if student_info else f"ID: {student_id}"
                            else:
                                afiseaza_mesaj("Scanati cardul!")
                                color = (0, 0, 255)  # Roșu pentru necunoscut
                                label = "Necunoscut"
                                
                                # Încearcă citirea NFC pentru persoanele necunoscute
                                logging.info("Persoană necunoscută detectată. Se încearcă citirea NFC...")
                                uid = read_nfc_uid()
                                if uid:
                                    student = find_student_by_nfc(uid)
                                    if student and student[0] not in studenti_marcati:
                                        threading.Thread(target=mark_attendance_local, 
                                                       args=(student[0], "nfc")).start()
                                        studenti_marcati.add(student[0])
                                        nume_complet = f"{student[2]} {student[1]}"
                                        label = f"{nume_complet} (NFC)"
                                        color = (255, 255, 0)  # Galben pentru NFC
                                        afiseaza_mesaj(nume_complet)
                        else:
                            afiseaza_mesaj("Student neidentificat!")
                            color = (0, 0, 255)
                            label = "Necunoscut"

                        # Desenează dreptunghiul și eticheta
                        cv2.rectangle(frame, (face.left(), face.top()), 
                                    (face.right(), face.bottom()), color, 2)
                        cv2.putText(frame, label, (face.left(), face.top() - 10), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

                frame_count += 1
                cv2.imshow("Sistem Prezență - Recunoaștere Facială", frame)
                
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

    except KeyboardInterrupt:
        logging.info("Sistem oprit de utilizator")
    except Exception as e:
        logging.error(f"Eroare în bucla principală: {e}")
    finally:
        # Curățare resurse
        if use_picamera:
            camera.close()
        else:
            cap.release()
        cv2.destroyAllWindows()
        logging.info("Sistem oprit și resurse curățate.")

if __name__ == "__main__":
    main()
