import threading
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
import numpy as np
import cv2
import imutils
import time
from datetime import datetime

lock = threading.Lock()
from app import mysql


def detect_and_predict_mask(frame, faceNet, maskNet):

    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300),
                                 (104.0, 177.0, 123.0))


    faceNet.setInput(blob)

    detections = faceNet.forward()

    faces = []
    locs = []
    preds = []

    # loop over the detections
    for i in range(0, detections.shape[2]):
        # extract the confidence (i.e., probability) associated with
        # the detection
        confidence = detections[0, 0, i, 2]

        # filter out weak detections by ensuring the confidence is
        # greater than the minimum confidence
        if confidence > 0.5:
            # compute the (x, y)-coordinates of the bounding box for
            # the object
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")

            # ensure the bounding boxes fall within the dimensions of
            # the frame
            (startX, startY) = (max(0, startX), max(0, startY))
            (endX, endY) = (min(w - 1, endX), min(h - 1, endY))

            # extract the face ROI, convert it from BGR to RGB channel
            # ordering, resize it to 224x224, and preprocess it
            face = frame[startY:endY, startX:endX]
            face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
            face = cv2.resize(face, (224, 224))
            face = img_to_array(face)
            face = preprocess_input(face)

            # add the face and bounding boxes to their respective
            # lists
            faces.append(face)
            locs.append((startX, startY, endX, endY))

    # only make a predictions if at least one face was detected
    if len(faces) > 0:
        # for faster inference we'll make batch predictions on *all*
        # faces at the same time rather than one-by-one predictions
        # in the above `for` loop
        faces = np.array(faces, dtype="float32")

        preds = maskNet.predict(faces, batch_size=32)


    return locs, preds


class Stream:

    def __init__(self, camera_src):
        self.camera_src = camera_src
        self.camera = None
        self.fps = 0
        self.amount_detected = 0
        self.withMask = 0
        self.withoutMask = 0


        self.maskFrames = 0
        self.noMaskFrames = 0

    def close(self):
        if self.camera is not None:
            self.camera.release()
            self.camera = None
            savePieChartData(self.maskFrames, self.noMaskFrames)

    def open(self):

        self.camera = cv2.VideoCapture(
            self.camera_src)

    def status(self):
        return self.camera is not None

    def get_fps(self):
        return self.fps

    def generateFrames(self, faceNet, maskNet):

        while True:
            if self.camera is not None:
                ret, frame = self.camera.read()
                start = time.time()

                if not ret:
                    break



                # detect faces in the frame and determine if they are wearing a
                # face mask or not
                (locs, preds) = detect_and_predict_mask(frame, faceNet, maskNet)

                # loop over the detected face locations and their corresponding
                # locations
                self.amount_detected = 0
                self.withMask = 0
                self.withoutMask = 0

                for (box, pred) in zip(locs, preds):
                    # unpack the bounding box and predictions
                    (startX, startY, endX, endY) = box
                    (mask, withoutMask) = pred

                    self.amount_detected += 1
                    if mask > withoutMask:
                        self.maskFrames += 1
                        self.withMask += 1

                    else:
                        self.noMaskFrames += 1
                        self.withoutMask += 1

                    # determine the class label and color we'll use to draw
                    # the bounding box and text
                    label = "Mask" if mask > withoutMask else "No Mask"
                    color = (0, 255, 0) if label == "Mask" else (0, 0, 255)

                    # include the probability in the label
                    label = "{}: {:.2f}%".format(label, max(mask, withoutMask) * 100)

                    # display the label and bounding box rectangle on the output
                    # frame
                    cv2.putText(frame, label, (startX, startY - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
                    cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)

                self.fps = 1.0 / (time.time() - start)




                (flag, encodedImage) = cv2.imencode(".jpg", frame)
                yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + bytearray(encodedImage) + b'\r\n')

# PieChart in die Datenbank übernehmen
def savePieChartData(maskFrames, noMaskFrames):
    try:

        cur = mysql.connection.cursor()
        result = cur.execute(
            "SELECT * FROM maskproportion WHERE date = CURDATE()")

        # wenn wir ein Ergebnis auf der Abfrage bekommen benutzen wir es ansonsten pushen wir ein neues ergebnis in die datenbank
        if result > 0:
            re = cur.fetchall()
            # die alten Frames zu den neuen hinzufügen
            updatedMaskFrames = maskFrames + re[0]["maskFrames"]
            updatedNoMaskFrames = noMaskFrames + re[0]["noMaskFrames"]

            cur.execute("UPDATE maskproportion SET maskFrames=%s, noMaskFrames=%s WHERE date=%s",
                        (updatedMaskFrames, updatedNoMaskFrames, datetime.now().strftime('%Y-%m-%d')))

            mysql.connection.commit()

            # Close connection
            cur.close()

        else:
            cur.execute("INSERT INTO maskproportion(date,maskFrames,noMaskFrames) VALUES(%s, %s, %s)",
                        (datetime.now().strftime('%Y-%m-%d'), maskFrames, noMaskFrames))

            # Commit to DB
            mysql.connection.commit()

            # Close connection
            cur.close()


    except ValueError:
        print("Value Error beim speichern vom pie chart!")
