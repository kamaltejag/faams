from imutils.video import VideoStream
import numpy as np
import cv2
import os
import time
import imutils
import json
import requests

from pathlib import Path

# For playing audio
# from gtts import gTTS 
# from playsound import playsound 

# Default Files Required for Face Detection
prototxt = "./face_detection_model/deploy.prototxt"
model = "./face_detection_model/res10_300x300_ssd_iter_140000.caffemodel"
min_confidence = 0.5

# REST API Url
url = 'http://faams.web:8000/rest_api/api/student/create.php'

img_id = 1

def save_image(img, user_id, img_id):
    Path("dataset/{}".format(user_id)).mkdir(parents=True, exist_ok=True)
    cv2.imwrite("dataset/{}/{}.jpg".format(user_id, img_id), img)
    time.sleep(1)

with open('students.json') as f:
    students = json.load(f)

# audios = ['./audios/look.mp3', './audios/tilt-left.mp3', './audios/tilt-right.mp3']
audios = ["Please look at the camera", "Please tilt your head to the left", "Please tilt your head to the right"]

print("[INFO] Loading Face Detector...")
net = cv2.dnn.readNetFromCaffe(prototxt, model)

print("[INFO] Starting video stream...")
vs = VideoStream(src=0).start()
time.sleep(2)

while True:
    print("[INFO] Enter the roll number of the student ")
    user_id = input().lower()
    check = False
    for student in students:
        if user_id in student.values():
            check = True
    if check:
        print("[ERROR] Roll Number already present")
        continue
    else:
        break
print("[Info] Enter the name of the student")
user_name = input().title()
print("[Info] Enter the branch of the student")
branch = input().upper()

print("[INFO] Capturing Images for {}...".format(user_id))

for i in range(0, 3):
    # playsound(audios[i])
    # playsound("./audios/countdown.mp3")
    # playsound("./audios/3.mp3")
    # playsound("./audios/2.mp3")
    # playsound("./audios/1.mp3")

    print("", format(audios[i]))
    time.sleep(2)

    for i in range(0, 3):
        frame = vs.read()
        frame = imutils.resize(frame, width=400)

        (h, w) = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(cv2.resize(
            frame, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))

        net.setInput(blob)
        detections = net.forward()

        if len(detections) > 0:
            for i in range(0, detections.shape[2]):
                confidence = detections[0, 0, i, 2]

                if confidence > min_confidence:
                    box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                    (startX, startY, endX, endY) = box.astype("int")

                    face = frame[startY:endY, startX:endX]

            save_image(face, user_id, img_id)
            img_id += 1

print("[INFO] Capturing Completed...")
# playsound("./audios/thank_you.mp3")


# Adding name and roll number to local json file
student = {}
student['roll_number'] = user_id
student['name'] = user_name
student['branch'] = branch
students.append(student)
with open('students.json', 'w') as f:
    json.dump(students, f)

print("[INFO] Stopping Video Stream...")
cv2.destroyAllWindows()
vs.stop()

print("[INFO] Sending data to server")
url = 'http://faams.web:8000/rest_api/api/student/create.php'
headers = {'Accept' : 'application/json', 'Content-Type' : 'application/json'}
r = requests.post(url, data=open('students.json', 'rb'), headers=headers)
data = r.json()
print("[INFO] {}".format(data['message']))