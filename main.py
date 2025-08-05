from ultralytics import YOLO
import cv2
import numpy as np
import face_recognition
import os
from datetime import datetime
import shutil
import time
import pyttsx3
import json

# === Paths ===
images_path = 'images'
backup_path = r'C:\Users\goyal\My Drive\FaceAttendanceSystem\FaceAttendanceBackup'
attendance_file = 'Attendance.csv'
mute_file = 'mute_state.json'
model_path = 'model/yolov8n-face-lindevs.pt'  # YOLOv8 face model

# === Voice Engine ===
voice_engine = pyttsx3.init()

def is_muted():
    try:
        with open(mute_file, 'r') as f:
            return json.load(f).get('muted', False)
    except:
        return False

def greet(name):
    if not is_muted():
        voice_engine.say(f"Hello, {name}")
        voice_engine.runAndWait()

# === Load and Encode Known Faces ===
def load_known_faces():
    images = []
    names = []
    if not os.path.exists(images_path):
        os.makedirs(images_path)

    for img_name in os.listdir(images_path):
        img_path = os.path.join(images_path, img_name)
        img = cv2.imread(img_path)
        if img is not None:
            images.append(img)
            names.append(os.path.splitext(img_name)[0])

    encodeList = []
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)
        if encode:
            encodeList.append(encode[0])

    return encodeList, names

# === Sync to Google Drive ===
def syncToDrive():
    try:
        dest = os.path.join(backup_path, attendance_file)
        shutil.copyfile(attendance_file, dest)
        time.sleep(1.5)
        print('☁️ Synced to Google Drive successfully')
    except Exception as e:
        print(f'⚠️ Sync failed: {e}')

# === Mark Attendance ===
def markAttendance(name):
    name = name.strip().upper()
    now = datetime.now()
    timeNow = now.strftime('%H:%M:%S')
    dateNow = now.strftime('%Y-%m-%d')

    if not os.path.exists(attendance_file):
        with open(attendance_file, 'w', encoding='utf-8') as f:
            f.write('Name,Time,Date\n')

    with open(attendance_file, 'r+', encoding='utf-8') as f:
        data = f.readlines()
        nameList = [line.split(',')[0].strip().upper() for line in data[1:]]
        if name not in nameList:
            f.write(f'{name},{timeNow},{dateNow}\n')
            f.flush()
            print(f'📌 Attendance marked for {name}')
            greet(name)
            syncToDrive()

# === Main Logic ===
def run_attendance():
    print('🔄 Loading known faces...')
    encodeListKnown, names = load_known_faces()
    print(f'✅ Loaded {len(encodeListKnown)} face(s):', names)

    # Load YOLOv8 face detection model
    model = YOLO(model_path)

    cap = cv2.VideoCapture(0)

    while True:
        success, frame = cap.read()
        if not success:
            break

        # Run YOLO detection (confidence threshold applied inside loop)
        results = model(frame, verbose=False)[0]

        for box in results.boxes:
            conf = float(box.conf[0])
            if conf < 0.7:
                continue

            x1, y1, x2, y2 = map(int, box.xyxy[0])
            face_roi = frame[y1:y2, x1:x2]

            if face_roi.size == 0:
                continue

            rgb_face = cv2.cvtColor(face_roi, cv2.COLOR_BGR2RGB)
            encodings = face_recognition.face_encodings(rgb_face)

            if encodings:
                encodeFace = encodings[0]
                matches = face_recognition.compare_faces(encodeListKnown, encodeFace)
                faceDis = face_recognition.face_distance(encodeListKnown, encodeFace)
                matchIndex = np.argmin(faceDis)

                if matches[matchIndex] and faceDis[matchIndex] < 0.5:
                    name = names[matchIndex]

                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.rectangle(frame, (x1, y2 - 35), (x2, y2), (0, 255, 0), cv2.FILLED)
                    cv2.putText(frame, name.upper(), (x1 + 6, y2 - 6),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                    markAttendance(name)

        cv2.imshow('Webcam - Face Attendance System', frame)

        if cv2.waitKey(1) == 27:  # ESC to quit
            break

    cap.release()
    cv2.destroyAllWindows()

# === Start Script ===
if __name__ == "__main__":
    run_attendance()
