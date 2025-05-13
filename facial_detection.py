import os
import cv2
import dlib
import numpy as np
import warnings
from glob import glob

# Suprimă warning-urile
os.environ["OPENCV_VIDEOIO_DEBUG"] = "0"
os.environ["GST_DEBUG"] = "0"
warnings.filterwarnings("ignore")

print("[INFO] Începem inițializarea...")

# Verifică dacă modelele există
if not os.path.exists("shape_predictor_68_face_landmarks.dat") or not os.path.exists("dlib_face_recognition_resnet_model_v1.dat"):
    print("[EROARE] Fisierele .dat lipsesc! Asigură-te că sunt în același folder cu scriptul.")
    exit()

# Inițializare Dlib
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
recognizer = dlib.face_recognition_model_v1("dlib_face_recognition_resnet_model_v1.dat")

faces = []
label_names = {}

print("[INFO] Începem încărcarea imaginilor din folderul 'poze/'...")

# Procesare imagini
for idx, filename in enumerate(glob("poze/*.*")):
    img = cv2.imread(filename)
    if img is None:
        print(f"[AVERTISMENT] Imagine invalidă sau coruptă: {filename}")
        continue

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    detected_faces = detector(gray)
    if len(detected_faces) == 0:
        print(f"[AVERTISMENT] Nicio față detectată în: {filename}")
        continue

    shape = predictor(img, detected_faces[0])  # trimitem imaginea color
    face_descriptor = recognizer.compute_face_descriptor(img, shape)

    name = os.path.splitext(os.path.basename(filename))[0]
    faces.append(face_descriptor)
    label_names[idx] = name
    print(f"[INFO] Imagine procesată: {filename} -> Etichetă: {name}")

print("[INFO] Accesăm camera video...")

# Deschide camera
cap = cv2.VideoCapture("/dev/video0", cv2.CAP_V4L2)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)
cap.set(cv2.CAP_PROP_FPS, 10)

if not cap.isOpened():
    print("[EROARE] Nu se poate deschide camera.")
    exit()

frame_count = 0
process_every_n_frames = 5

while True:
    ret, frame = cap.read()
    if not ret:
        print("[EROARE] Nu s-a putut citi frame-ul.")
        break

    frame = cv2.flip(frame, 1)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    if frame_count % process_every_n_frames == 0:
        detected_faces = detector(gray)

        for face in detected_faces:
            shape = predictor(frame, face)  # trimitem imaginea color
            face_descriptor = recognizer.compute_face_descriptor(frame, shape)

            distances = [np.linalg.norm(np.array(face_descriptor) - np.array(f)) for f in faces]
            min_dist = min(distances) if distances else float("inf")
            id_ = distances.index(min_dist) if distances else -1

            name = label_names.get(id_, "Necunoscut") if min_dist < 0.6 else "Necunoscut"
            color = (0, 255, 0) if name != "Necunoscut" else (0, 0, 255)

            cv2.rectangle(frame, (face.left(), face.top()), (face.right(), face.bottom()), color, 2)
            cv2.putText(frame, f"{name} ({round(min_dist, 2)})", (face.left(), face.top() - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            print(f"[DEBUG] Față detectată: {name} | Distanță: {min_dist:.4f}")

    frame_count += 1
    cv2.imshow('Camera Live - Face Recognition', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("[INFO] Ieșire din program.")
        break

cap.release()
cv2.destroyAllWindows()
cv2.waitKey(1)
