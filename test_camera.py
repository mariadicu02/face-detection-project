import cv2
cap = cv2.VideoCapture(0)
ret, frame = cap.read()
if ret:
    cv2.imwrite('test_frame.jpg', frame)
cap.release()
print("Frame salvat în test_frame.jpg")
