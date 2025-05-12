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

# Inițializează detectorul de fețe Dlib
detector = dlib.get_frontal_face_detector()
# Inițializează predictorul de puncte faciale
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
# Inițializează modelul de recunoaștere facială Dlib
recognizer = dlib.face_recognition_model_v1("dlib_face_recognition_resnet_model_v1.dat")

faces = []
label_names = {}

print("[INFO] Începem încărcarea imaginilor din folderul 'poze/'...")

for idx, filename in enumerate(glob("poze/*.*")):
    img = cv2.imread(filename)
    
    detected_faces = detector(img)

    if len(detected_faces) == 0:
        print(f"[AVERTISMENT] Nu s-a detectat nicio față în: {filename}")
        continue

    shape = predictor(img, detected_faces[0])
    face_descriptor = recognizer.compute_face_descriptor(img, shape)

    name = os.path.splitext(os.path.basename(filename))[0]
    faces.append(face_descriptor)
    label_names[idx] = name
    print(f"[INFO] Imagine procesată: {filename} -> Etichetă: {name}")

print("[INFO] Accesăm camera video...")
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("[EROARE] Nu se poate deschide camera.")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("[EROARE] Nu s-a putut citi frame-ul.")
        break

    frame = cv2.flip(frame, 1)
    detected_faces = detector(frame)

    for face in detected_faces:
        shape = predictor(frame, face)
        face_descriptor = recognizer.compute_face_descriptor(frame, shape)

        distances = [np.linalg.norm(np.array(face_descriptor) - np.array(f)) for f in faces]
        min_dist = min(distances)
        id_ = distances.index(min_dist)

        name = label_names.get(id_, "Necunoscut") if min_dist < 0.6 else "Necunoscut"
        color = (0, 255, 0) if min_dist < 0.6 else (0, 0, 255)

        cv2.rectangle(frame, (face.left(), face.top()), (face.right(), face.bottom()), color, 2)
        cv2.putText(frame, f"{name} ({round(min_dist, 2)})", (face.left(), face.top() - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

    cv2.imshow('Camera Live - Face Recognition', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("[INFO] Ieșire din program.")
        break

cap.release()
cv2.destroyAllWindows()
cv2.waitKey(1)
