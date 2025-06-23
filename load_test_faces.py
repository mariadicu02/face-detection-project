import cv2
import numpy as np
import glob
from pathlib import Path
import dlib

# 1. Inițializare Dlib
detector   = dlib.get_frontal_face_detector()
predictor  = dlib.shape_predictor("models/shape_predictor_68_face_landmarks.dat")
recognizer = dlib.face_recognition_model_v1("models/dlib_face_recognition_resnet_model_v1.dat")

def load_test_faces():
    known_descriptors = []
    known_names       = []
    for file in glob.glob("poze_moodle/*.jpg"):
        name = Path(file).stem  
        img  = cv2.imread(file)

        gray  = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = detector(gray)
        if not faces:
            print(f"Nicio față detectată în {name}")
            continue


        shape = predictor(gray, faces[0])
        desc  = np.array(recognizer.compute_face_descriptor(img, shape))

        known_descriptors.append(desc)
        known_names.append(name)
        print(f"Încărcat descriptor pentru {name}")

    print(f"Încărcate {len(known_descriptors)} descriptori AI")
    return known_descriptors, known_names

if __name__ == "__main__":
    load_test_faces()
