from imutils.video import VideoStream
from imutils.video import FPS
import numpy as np
import imutils
import pickle
import time
import cv2
import os
import datetime
import json
import requests

from gtts import gTTS 
from playsound import playsound 


# from functions import *

# Initialize required files
dataset = "dataset"
prototxt = "./face_detection_model/deploy.prototxt"
model = "./face_detection_model/res10_300x300_ssd_iter_140000.caffemodel"
min_confidence = 0.5
embeddings_model = "openface_nn4.small2.v1.t7"
recognizer = "./output/recognizer.pickle"
le = "./output/le.pickle"

print("[INFO] Loading face detector...")
detector = cv2.dnn.readNetFromCaffe(prototxt, model)

print("[INFO] Loading face recognizer...")
embedder = cv2.dnn.readNetFromTorch(embeddings_model)

recognizer = pickle.loads(open(recognizer, "rb").read())
le = pickle.loads(open(le, "rb").read())

print("[INFO] Starting video stream...")
vs = VideoStream(src=-1).start()
time.sleep(2.0)

date = datetime.datetime.now()
print("[INFO] Generating list to keep track of attendance on {}".format(date))

while True:
	frame = vs.read()

	# resize the frame to have a width of 600 pixels (while
	# maintaining the aspect ratio), and then grab the image
	# dimensions
	frame = imutils.resize(frame, width=600)
	(h, w) = frame.shape[:2]

	# construct a blob from the image
	imageBlob = cv2.dnn.blobFromImage(
		cv2.resize(frame, (300, 300)), 1.0, (300, 300),
		(104.0, 177.0, 123.0), swapRB=False, crop=False)

	# apply OpenCV's deep learning-based face detector to localize
	# faces in the input image
	detector.setInput(imageBlob)
	detections = detector.forward()

	# loop over the detections
	for i in range(0, detections.shape[2]):
		# extract the confidence (i.e., probability) associated with
		# the prediction
		confidence = detections[0, 0, i, 2]

		# filter out weak detections
		if confidence > min_confidence:
			# compute the (x, y)-coordinates of the bounding box for
			# the face
			box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
			(startX, startY, endX, endY) = box.astype("int")

			# extract the face ROI
			face = frame[startY:endY, startX:endX]
			(fH, fW) = face.shape[:2]

			# print("X: {}, Y: {}".format(fW, fH))

			y = startY - 10 if startY - 10 > 10 else startY + 10
			cv2.rectangle(frame, (startX, startY), (endX, endY),
				(0, 0, 255), 2)

			# ensure the face width and height are sufficiently large
			if fW < 110 or fH < 140:
				continue
			elif fW < 120 or fH < 150:
				playsound("./audios/move_closer.mp3")
				time.sleep(3)
				continue
			elif fW > 150 or fH > 190:
				playsound("./audios/move_back.mp3")
				time.sleep(3)
				continue

			# construct a blob for the face ROI, then pass the blob
			# through our face embedding model to obtain the 128-d
			# quantification of the face
			faceBlob = cv2.dnn.blobFromImage(face, 1.0 / 255,
				(96, 96), (0, 0, 0), swapRB=True, crop=False)
			embedder.setInput(faceBlob)
			vec = embedder.forward()

			# perform classification to recognize the face
			preds = recognizer.predict_proba(vec)[0]
			j = np.argmax(preds)
			proba = preds[j]
			roll_number = le.classes_[j]


			with open('attendance.json') as f:
				attendance = json.load(f)

			# Check if attendance already exists
			check = True
			for item in attendance:
				if roll_number in item.values():
					check = False
			if check:
				# Add date and roll number to dictionary
				student_attendance = {}
				student_attendance['date'] = date.strftime("%d-%m-%Y")
				student_attendance['roll_number'] = roll_number
				attendance.append(student_attendance)
				with open('attendance.json', 'w') as f:
					json.dump(attendance, f)

				# Add attendance to database and retrieve name
				url = 'http://faams.web:8000/rest_api/api/attendance/create.php'
				headers = {'Accept' : 'application/json', 'Content-Type' : 'application/json'}
				r = requests.post(url, data=json.dumps(student_attendance), headers=headers)
				r = r.json()

				with open('students.json') as f:
					students = json.load(f)

				name = ""
				
				for student in students:
					if roll_number in student.values():
						name = student.get('name')

				print("[INFO] Attendance added for {}".format(name))
				# print(students)

				text = "Good morning {}, your attendance has been recorded. Please enter the classroom".format(name) 
				var = gTTS(text = text,lang = 'en') 
				var.save('./audios/confirm.mp3') 
				playsound("./audios/confirm.mp3")
				time.sleep(5)

	# show the output frame
	cv2.imshow("Frame", frame)
	key = cv2.waitKey(1) & 0xFF

	# if the `q` key was pressed, break from the loop
	if key == ord("q"):
		break
	
cv2.destroyAllWindows()
vs.stop()