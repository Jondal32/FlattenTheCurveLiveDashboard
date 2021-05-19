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
        self.fps = 0
        self.amount_detected = 0
        self.MIN_DISTANCE = 50
        self.configPath = "models/yolov3/yolov3.cfg"
        self.weightsPath = "models/yolov3/yolov3.weights"
        self.net = cv2.dnn.readNetFromDarknet(self.configPath, self.weightsPath)
        self.MIN_CONF = 0.3
        self.NMS_THRESH = 0.3
        self.USE_GPU = False

        if self.USE_GPU:
            # set CUDA as the preferable backend and target
            print("[INFO] setting preferable backend and target to CUDA...")
            self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
            self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

        self.ln = self.net.getLayerNames()
        self.ln = [self.ln[i[0] - 1] for i in self.net.getUnconnectedOutLayers()]

        # info for cards
        #
        self.fps = 0
        self.violations = 0
        self.amount_detected = 0
        self.index = 0

        self.violations_list = MaxSizeList(50)

    def close(self):
        if self.camera is not None:
            self.camera.release()
            self.camera = None
            self.violations_list.list_to_csv()

    def open(self):
        self.camera = cv2.VideoCapture(self.camera_src)

    def status(self):
        return self.camera is not None

    def get_fps(self):
        return self.fps

    def generateFrames(self):
        while True:
            (grabbed, frame) = self.camera.read()
            self.index += 1
            if self.index % 60 != 0:
                continue
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
                    # alle Verstöße zur Liste hinzufügen die zur Erstellung der HeatMap geeignet ist
                    self.violations_list.push([int(startX), int(startY)])

                # draw (1) a bounding box around the person and (2) the
                # centroid coordinates of the person,
                thickness = 2
                cv2.rectangle(frame, (int(startX), int(startY)), (int(endX), int(endY)), color, thickness)
                cv2.circle(frame, (int(cX), int(cY)), 5, color, 1)

            self.violations = len(violate)
            self.amount_detected = len(results)

            # draw the total number of social distancing violations on the
            # output frame

            # check to see if the output frame should be displayed to our
            # screen
            self.fps = 1.0 / (time.time() - start)
            (flag, encodedImage) = cv2.imencode(".jpg", frame)
            yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + bytearray(encodedImage) + b'\r\n')



    def detect_people(self, frame):
        (H, W) = frame.shape[:2]
        results = []

        # construct a blob from the input frame and then perform a forward
        # pass of the YOLO object detector, giving us our bounding boxes
        # and associated probabilities
        blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (416, 416),
                                     swapRB=True, crop=False)

        self.net.setInput(blob)
        layerOutputs = self.net.forward(self.ln)
        detections = self.net.forward()

        # detect(frame, detections)

        # initialize our lists of detected bounding boxes, centroids, and
        # confidences, respectively
        boxes = []
        centroids = []
        confidences = []

        # loop over each of the layer outputs
        for output in layerOutputs:
            # loop over each of the detections
            for detection in output:

                # extract the class ID and confidence (i.e., probability)
                # of the current object detection
                scores = detection[5:]
                classID = np.argmax(scores)
                confidence = scores[classID]

                # filter detections by (1) ensuring that the object
                # detected was a person and (2) that the minimum
                # confidence is met
                if classID == 0 and confidence > self.MIN_CONF:
                    # scale the bounding box coordinates back relative to
                    # the size of the image, keeping in mind that YOLO
                    # actually returns the center (x, y)-coordinates of
                    # the bounding box followed by the boxes' width and
                    # height
                    box = detection[0:4] * np.array([W, H, W, H])
                    (centerX, centerY, width, height) = box.astype("int")

                    # use the center (x, y)-coordinates to derive the top
                    # and and left corner of the bounding box
                    x = int(centerX - (width / 2))
                    y = int(centerY - (height / 2))

                    # update our list of bounding box coordinates,
                    # centroids, and confidences
                    boxes.append([x, y, int(width), int(height)])
                    centroids.append((centerX, centerY))
                    confidences.append(float(confidence))

        # apply non-maxima suppression to suppress weak, overlapping
        # bounding boxes
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
        with open('static/img/output.csv', 'w') as f:
            writer = csv.writer(f)
            writer.writerow(['x', 'y'])
            writer.writerows(self.ls)
