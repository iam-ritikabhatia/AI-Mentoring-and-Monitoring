import cv2
import numpy as np
import face_recognition
import os
from datetime import datetime
import serial
import requests

arduino = serial.Serial('COM6', 9600)

marked_names = set()

def get_esp32cam_image(esp32cam_url):
    try:
        response = requests.get(esp32cam_url, timeout=10)
        if response.status_code == 200:
            img_array = np.array(bytearray(response.content), dtype=np.uint8)
            img = cv2.imdecode(img_array, -1)
            return img
    except Exception as e:
        print(f"Error fetching image from ESP32-CAM: {str(e)}")
    return None


def find_encodings(images):
    encodeList = []
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]
        encodeList.append(encode)
    return encodeList

def mark_attendance(name, roll_number, section):
    if name not in marked_names:
        with open('Attendance.csv', 'a') as f:
            now = datetime.now()
            dtString = now.strftime('%H:%M:%S')
            f.write(f'{name},{roll_number},{section},{dtString}\n')
            print(f'{name} {roll_number} {section} {dtString}')
            marked_names.add(name)
            arduino.write(f'{name},{roll_number},{section}'.encode())

path = 'ImagesBasic'
images = []
classNames = []
mylist = os.listdir(path)
for cl in mylist:
    if cl.endswith(('.jpg', '.jpeg', '.png', '.gif')):
        curImg = cv2.imread(os.path.join(path, cl))
        if curImg is not None:
            images.append(curImg)
            # Extracting components from the file name
            components = os.path.splitext(cl)[0].split('_')
            name = components[0]
            roll_number = components[1]
            section = components[2]
            print(f"Name: {name}, Roll Number: {roll_number}, Section: {section}")  # Debugging print statement
            classNames.append((name, roll_number, section))

encodelistknown = find_encodings(images)
print('Encoding Complete!')

esp32cam_url = 'http://192.168.201.68:8080/video'

attendance_status = {}
previous_names = set()

while True:
    img = get_esp32cam_image(esp32cam_url)

    if img is not None:
        imgS = cv2.resize(img, (0, 0), None, 0.25, 0.25)
        imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)

        faceCurFrame = face_recognition.face_locations(imgS)
        encodesCurFrame = face_recognition.face_encodings(imgS, faceCurFrame)

        current_names = set()

        for encodeFace, faceLoc in zip(encodesCurFrame, faceCurFrame):
            matches = face_recognition.compare_faces(encodelistknown, encodeFace)
            faceDis = face_recognition.face_distance(encodelistknown, encodeFace)
            matchIndex = np.argmin(faceDis)

            if matches[matchIndex]:
                name, roll_number, section = classNames[matchIndex]
                current_names.add(name)

                mark_attendance(name, roll_number, section)

                y1, x2, y2, x1 = faceLoc
                y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4
                if name in previous_names:
                    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(img, "Marked", (x1 + 6, y2 - 6), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
                else:
                    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.rectangle(img, (x1, y2 - 35), (x2, y2), (255, 255, 0), cv2.FILLED)
                    cv2.putText(img, f'{name} ({roll_number}, {section})', (x1 + 6, y2 - 6), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 0), 2)

        previous_names = current_names

        cv2.imshow('ESP32-CAM', img)
        cv2.waitKey(1)

cv2.destroyAllWindows()
