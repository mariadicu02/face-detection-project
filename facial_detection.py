import os
import cv2
import numpy as np
import warnings

# Suprimă warning-urile OpenCV și GStreamer
os.environ["OPENCV_VIDEOIO_DEBUG"] = "0"
os.environ["GST_DEBUG"] = "0"
warnings.filterwarnings("ignore")

print("[INFO] Începem inițializarea...")

# Încarcă clasificatorul Haar pentru față
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Creează recognizer facial LBPH
recognizer = cv2.face.LBPHFaceRecognizer_create()

faces = []
labels = []
label_names = {}
current_id = 0

print("[INFO] Începem încărcarea imaginilor din folderul 'poze/'...")

for filename in os.listdir("poze"):
    if filename.lower().endswith((".jpg", ".png", ".jpeg")):
        path = os.path.join("poze", filename)
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            print(f"[AVERTISMENT] Nu s-a putut încărca: {filename}")
            continue

        detected = face_cascade.detectMultiScale(img, scaleFactor=1.1, minNeighbors=5)

        if len(detected) == 0:
            print(f"[AVERTISMENT] Nu s-a detectat nicio față în: {filename}")
            continue

        for (x, y, w, h) in detected:
            roi = img[y:y+h, x:x+w]
            roi = cv2.resize(roi, (200, 200))  # Dimensiune standard
            roi = cv2.equalizeHist(roi)       # Normalizare contrast

            name = os.path.splitext(filename)[0]  # Ex: simona_vlad.jpg -> simona_vlad
            faces.append(roi)
            labels.append(current_id)
            label_names[current_id] = name
            current_id += 1
            print(f"[INFO] Imagine procesată: {filename} -> Etichetă: {name}")
            break  # Folosim doar prima față din fiecare imagine

print("[INFO] Antrenăm recognizer-ul facial...")
recognizer.train(faces, np.array(labels))
print("[INFO] Antrenare completă!")

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

    frame = cv2.flip(frame, 1)  # Efect de mirror
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces_detected = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

    for (x, y, w, h) in faces_detected:
        roi_gray = gray[y:y+h, x:x+w]
        roi_gray = cv2.resize(roi_gray, (200, 200))
        roi_gray = cv2.equalizeHist(roi_gray)

        id_, confidence = recognizer.predict(roi_gray)

        if confidence < 60:
            name = label_names.get(id_, "Necunoscut")
            label = f"{name} ({round(confidence, 2)})"
            color = (0, 255, 0)
        else:
            label = "Necunoscut"
            color = (0, 0, 255)

        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
        cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

    cv2.imshow('Camera Live - Face Recognition', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("[INFO] Ieșire din program.")
        break

cap.release()
cv2.destroyAllWindows()
