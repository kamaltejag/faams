from imutils import paths
import numpy as np
import imutils
import pickle
import cv2
import os

# For Training Model
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
# import pickle

dataset = "dataset"
prototxt = "./face_detection_model/deploy.prototxt"
model = "./face_detection_model/res10_300x300_ssd_iter_140000.caffemodel"
min_confidence = 0.5
embeddings_model = "openface_nn4.small2.v1.t7"

print("[INFO] Loading Face Detector...")
detector = cv2.dnn.readNetFromCaffe(prototxt, model)

print("[INFO] Loading Face Recognizer...")
embedder = cv2.dnn.readNetFromTorch(embeddings_model)

print("[INFO] Quantifying Faces...")
imagePaths = list(paths.list_images(dataset))

knownEmbeddings = []
knownNames = []

total = 0

# loop over the image paths
for (i, imagePath) in enumerate(imagePaths):
    # extract the person name from the image path
    print("[INFO] processing image {}/{}".format(i + 1,
        len(imagePaths)))
    name = imagePath.split(os.path.sep)[-2]

    face = cv2.imread(imagePath)
    faceBlob = cv2.dnn.blobFromImage(face, 1.0 / 255, (96, 96), (0, 0, 0), swapRB=True, crop=False)
    embedder.setInput(faceBlob)
    vec = embedder.forward()

    knownNames.append(name)
    knownEmbeddings.append(vec.flatten())
    total += 1

print(knownNames)
# dump the facial embeddings + names to disk
print("[INFO] Serializing {} Encodings...".format(total))
data = {"embeddings": knownEmbeddings, "names": knownNames}
# f = open("./output/embeddings.pickle", "wb")
# f.write(pickle.dumps(data))
# f.close()
print("[Info] Serialization Completed")


# load the face embeddings
# print("[INFO] Loading Face Embeddings...")
# data = pickle.loads(open("./output/embeddings.pickle", "rb").read())

# encode the labels
print("[INFO] Encoding Labels...")
le = LabelEncoder()
labels = le.fit_transform(data["names"])

# train the model used to accept the 128-d embeddings of the face and
# then produce the actual face recognition
print("[INFO] Training Model...")
recognizer = SVC(C=1.0, kernel="linear", probability=True)
recognizer.fit(data["embeddings"], labels)

# write the actual face recognition model to disk
f = open("./output/recognizer.pickle", "wb")
f.write(pickle.dumps(recognizer))
f.close()

# write the label encoder to disk
f = open("./output/le.pickle", "wb")
f.write(pickle.dumps(le))
f.close()
print("[Info] Training Completed")