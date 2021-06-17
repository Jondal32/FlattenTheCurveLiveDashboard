# import the necessary packages
from scipy.spatial import distance as dist
import numpy as np
import cv2
import time
import csv


class Stream:

    def __init__(self, camera_src):
        self.camera_src = camera_src
        self.camera = None
        self.MIN_DISTANCE = 75
        self.pbFile = pbFile = r"C:\Users\manue\PycharmProjects\einfaches_dashboard_feb_2021\models\ssd_mobilenet_v2_coco_2018_03_29\frozen_inference_graph.pb"
        self.pbtxtFile = r"C:\Users\manue\PycharmProjects\einfaches_dashboard_feb_2021\models\ssd_mobilenet_v2_coco_2018_03_29\ssd_mobilenet_v2_coco_2018_03_29.pbtxt"
        self.net = cv2.dnn.readNetFromTensorflow(self.pbFile, self.pbtxtFile)
        self.MIN_CONF = 0.5
        self.NMS_THRESH = 0.3
        self.violations_list = MaxSizeList(50)

        self.H = None
        self.W = None

        # info for cards
        self.fps = 0
        self.violations = 0
        self.amount_detected = 0

        # Width of network's input image
        self.inputWidth = 300
        # Height of network's input image
        self.inputHeight = 300

    def close(self):
        if self.camera is not None:
            self.camera.release()
            self.camera = None
            self.net = None
            self.violations_list.list_to_csv()

    def open(self):
        self.camera = cv2.VideoCapture(self.camera_src)
        self.net = cv2.dnn.readNetFromTensorflow(self.pbFile, self.pbtxtFile)

        self.H = int(self.camera.get(cv2.CAP_PROP_FRAME_HEIGHT))

        self.W = int(self.camera.get(cv2.CAP_PROP_FRAME_WIDTH))
        print(self.H, self.W)

    def status(self):
        return self.camera is not None

    def get_fps(self):
        return self.fps

    def generateFrames(self):
        while True:

            if self.camera is not None:
                (grabbed, frame) = self.camera.read()
                start = time.time()

                # if the frame was not grabbed, then we have reached the end
                # of the stream
                if not grabbed:
                    self.camera = cv2.VideoCapture(self.camera_src)

                    continue

                results = self.detect_people(frame)

                violate = set()

                # ensure there are *at least* two people detections (required in
                # order to compute our pairwise distance maps)
                if len(results) >= 2:
                    # extract all centroids from the results and compute the
                    # Euclidean distances between all pairs of the centroids
                    centroids = np.array([r[2] for r in results])
                    D = dist.cdist(centroids, centroids, metric="euclidean")

                    # loop over the upper triangular of the distance matrix
                    for i in range(0, D.shape[0]):
                        for j in range(i + 1, D.shape[1]):
                            # check to see if the distance between any two
                            # centroid pairs is less than the configured number
                            # of pixels
                            if D[i, j] < self.MIN_DISTANCE:
                                # update our violation set with the indexes of
                                # the centroid pairs
                                violate.add(i)
                                violate.add(j)

                # loop over the results
                for (i, (prob, bbox, centroid)) in enumerate(results):
                    # extract the bounding box and centroid coordinates, then
                    # initialize the color of the annotation
                    (startX, startY, endX, endY) = bbox
                    (cX, cY) = centroid
                    color = (0, 255, 0)

                    # if the index pair exists within the violation set, then
                    # update the color
                    if i in violate:
                        color = (0, 0, 255)
                        self.violations_list.push([int(cX), int(cY)])

                    # draw (1) a bounding box around the person and (2) the
                    # centroid coordinates of the person,
                    cv2.rectangle(frame, (int(startX), int(startY)), (int(endX), int(endY)), color, thickness=2)
                    cv2.circle(frame, (int(cX), int(cY)), 5, color, 1)

                # draw the total number of social distancing violations on the
                # output frame
                self.violations = len(violate)
                self.amount_detected = len(results)

                # check to see if the output frame should be displayed to our
                # screen
                self.fps = 1.0 / (time.time() - start)

                (flag, encodedImage) = cv2.imencode(".jpg", frame)
                yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + bytearray(encodedImage) + b'\r\n')
            else:
                if self.net is not None:
                    self.net = None

    def detect_people(self, frame):
        # grab the dimensions of the frame and  initialize the list of
        # results
        blob = cv2.dnn.blobFromImage(
            frame, size=(self.inputWidth, self.inputHeight), swapRB=True, crop=False
        )

        # Feed the input blob to the network, perform inference and get the output:
        # Set the input for the network
        self.net.setInput(blob)

        detections = self.net.forward()

        results = []

        boxes = []
        centroids = []
        confidences = []

        for detection in detections[0, 0, :, :]:
            confidence = float(detection[2])

            # if the confidence is above a threshold
            if confidence > self.MIN_CONF:
                classID = detection[1]

                # proceed only if the object detected is indeed a human
                if classID == 1:
                    # get coordinates of the bbox
                    left = detection[3] * self.W
                    top = detection[4] * self.H
                    right = detection[5] * self.W
                    bottom = detection[6] * self.H

                    width = right - left
                    height = bottom - top

                    boxes.append([left, top, width, height])
                    centroids.append((right - (left / 2), bottom - (top / 2)))
                    confidences.append(confidence)

        idxs = cv2.dnn.NMSBoxes(boxes, confidences, self.MIN_CONF, self.NMS_THRESH)

        # ensure at least one detection exists
        if len(idxs) > 0:
            # loop over the indexes we are keeping
            for i in idxs.flatten():
                # extract the bounding box coordinates
                (x, y) = (boxes[i][0], boxes[i][1])
                (w, h) = (boxes[i][2], boxes[i][3])

                # update our results list to consist of the person
                # prediction probability, bounding box coordinates,
                # and the centroid
                r = (confidences[i], (x, y, x + w, y + h), centroids[i])
                results.append(r)

        # return the list of results
        return results


class MaxSizeList(object):
    """
    Anstatt Daten in die Datenabank zu übertragen wird hier eine lokale Liste gespeichert und stetig aktualisiert mit den letzten X-Abstandsunterschreitungen
    """

    def __init__(self, max_length):
        self.max_length = max_length
        self.ls = []

    def push(self, st):
        if len(self.ls) == self.max_length:
            self.ls.pop(0)
        self.ls.append(st)

    def get_list(self):
        return self.ls

    def list_to_csv(self):
        with open('static/img/output.csv', 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['x', 'y'])
            writer.writerows(self.ls)
