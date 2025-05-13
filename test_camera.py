import cv2

# Încearcă să accesezi camera video
cap = cv2.VideoCapture("/dev/video0")  # sau 0 cu cv2.CAP_V4L2 dacă vrei

if not cap.isOpened():
    print("[EROARE] Nu s-a putut deschide camera.")
    exit()

print("[INFO] Camera deschisă cu succes. Apasă 'q' pentru a ieși.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("[EROARE] Nu s-a putut citi un cadru.")
        break

    cv2.imshow("Test Camera", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
