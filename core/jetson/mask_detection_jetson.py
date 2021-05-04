import threading
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
import numpy as np
import cv2
import imutils
import time
from datetime import datetime
import jetson.inference
import jetson.utils

lock = threading.Lock()
from app import mysql


def detect_and_predict_mask(frame, faceNet, maskNet):
    # grab the dimensions of the frame and then construct a blob
    # from it
    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300),
                                 (104.0, 177.0, 123.0))

    # pass the blob through the network and obtain the face detections
    faceNet.setInput(blob)
    detections = faceNet.forward()

    # initialize our list of faces, their corresponding locations,
    # and the list of predictions from our face mask network
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

    # return a 2-tuple of the face locations and their corresponding
    # locations
    return locs, preds


class Stream:

    def __init__(self, camera_src):
        self.camera_src = camera_src
        self.camera = None
        self.prev_messages = ""
        self.fps = 0
        self.amount_detected = 0
        self.withMask = 0
        self.withoutMask = 0


        self.maskFrames = 0
        self.noMaskFrames = 0

        self.stream_on = False
        self.net = jetson.inference.detectNet("ssd-mobilenet-v2", threshold=0.5)

        self.timeStamp = time.time()
        self.font = cv2.FONT_HERSHEY_COMPLEX

    def close(self):
        self.stream_on = False
        savePieChartData(self.maskFrames, self.noMaskFrames)

    def open(self):
        self.stream_on = True

    def status(self):
        return self.stream_on

    def get_fps(self):
        return self.fps

    def generateFrames(self, faceNet, maskNet):

        while True:
            if self.stream_on:
                ret, img = self.camera.read()
                if not ret:
                    self.camera = cv2.VideoCapture(self.camera_src)

                    continue

                height = img.shape[0]
                width = img.shape[1]
                # img = camera.Capture()

                # frame = cv2.cvtColor(img, cv2.COLOR_BGR2RGBA).astype(np.float32)
                frame = jetson.utils.cudaFromNumpy(img)

                detections = self.net.Detect(frame, width, height, overlay='none')
                for detect in detections:
                    # print(detect)
                    ID = detect.ClassID
                    top = detect.Top
                    left = detect.Left
                    bottom = detect.Bottom
                    right = detect.Right
                    item = self.net.GetClassDesc(ID)
                    # print(item, top, left, bottom, right)

                    img = cv2.rectangle(img,
                                        (int(left), int(top)),
                                        (int(right), int(bottom)), color=(0, 0, 255), thickness=2)
                # display.RenderOnce(img,width,height)
                dt = time.time() - self.timeStamp
                timeStamp = time.time()
                fps = 1 / dt
                self.fpsFilt = .9 * self.fpsFilt + .1 * fps
                # print(str(round(fps,1))+' fps')
                self.fps = round(self.fpsFilt, 1)
                cv2.putText(img, str(self.fps) + ' fps', (0, 30), self.font, 1, (0, 0, 255), 2)

                (flag, encodedImage) = cv2.imencode(".jpg", img)
                yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + bytearray(encodedImage) + b'\r\n')


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
