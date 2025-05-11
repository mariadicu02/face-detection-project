import os
import cv2
import dlib
import numpy as np
import warnings

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
labels = []
label_names = {}
current_id = 0

print("[INFO] Începem încărcarea imaginilor din folderul 'poze/'...")

for filename in os.listdir("poze"):
    if filename.lower().endswith((".jpg", ".png", ".jpeg")):
        path = os.path.join("poze", filename)
        img = cv2.imread(path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        detected_faces = detector(gray)

        if len(detected_faces) == 0:
            print(f"[AVERTISMENT] Nu s-a detectat nicio față în: {filename}")
            continue

        for face in detected_faces:
            shape = predictor(gray, face)  # Predictor facial adăugat
            face_descriptor = recognizer.compute_face_descriptor(img, shape)

            name = os.path.splitext(filename)[0]
            faces.append(face_descriptor)
            labels.append(current_id)
            label_names[current_id] = name
            current_id += 1
            print(f"[INFO] Imagine procesată: {filename} -> Etichetă: {name}")
            break

print("[INFO] Antrenamentul nu este necesar, Dlib folosește deep learning!")

# Pornim camera
print("[INFO] Accesăm camera video...")
cap = cv2.VideoCapture(0, cv2.CAP_V4L2)

if not cap.isOpened():
    print("[EROARE] Nu se poate deschide camera.")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("[EROARE] Nu s-a putut citi frame-ul.")
        break

    frame = cv2.flip(frame, 1)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    detected_faces = detector(gray)

    for face in detected_faces:
        shape = predictor(gray, face)  # Predictor facial adăugat
        face_descriptor = recognizer.compute_face_descriptor(frame, shape)

        distances = [np.linalg.norm(np.array(face_descriptor) - np.array(f)) for f in faces]
        min_dist = min(distances)
        id_ = distances.index(min_dist)

        if min_dist < 0.6:  # Prag pentru recunoaștere
            name = label_names.get(id_, "Necunoscut")
            label = f"{name} ({round(min_dist, 2)})"
            color = (0, 255, 0)
        else:
            label = "Necunoscut"
            color = (0, 0, 255)

        cv2.rectangle(frame, (face.left(), face.top()), (face.right(), face.bottom()), color, 2)
        cv2.putText(frame, label, (face.left(), face.top() - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

    cv2.imshow('Camera Live - Face Recognition', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("[INFO] Ieșire din program.")
        break

cap.release()
cv2.destroyAllWindows()
