import cv2
import time
import numpy as np
from scipy.spatial import distance as dist
from collections import OrderedDict

# used for tracking
import dlib

inputFile = r"C:\Users\manue\PycharmProjects\einfaches_dashboard_feb_2021\videos\1.mp4"
# outputFile = "../../../data/output tf/pi22_tf_inc_10.avi"


# minimum probability to filter weak detections
minConfidence = 0.5

# switch between detection and tracking
# set number of frames to skip before doing a detection
skipFrames = 5

FPSUpdate = 50

# Width of network's input image
inputWidth = 300
# Height of network's input image
inputHeight = 300

font = cv2.FONT_HERSHEY_SIMPLEX

pbFile = r"C:\Users\manue\PycharmProjects\einfaches_dashboard_feb_2021\models\ssd_mobilenet_v2_coco_2018_03_29\frozen_inference_graph.pb"
pbtxtFile = r"C:\Users\manue\PycharmProjects\einfaches_dashboard_feb_2021\models\ssd_mobilenet_v2_coco_2018_03_29\ssd_mobilenet_v2_coco_2018_03_29.pbtxt"


modelName = "SSD Inception V2"

# cv2.dnn.writeTextGraph(pbFile, 'graph.pbtxt')

status = "off"

# WRITER to save computed stream to device
writer = None

vs = cv2.VideoCapture(inputFile)

# get total frame to compute remaining processing time
prop = cv2.CAP_PROP_FRAME_COUNT
totalFrames = int(vs.get(prop))

#H = int(vs.get(cv2.CAP_PROP_FRAME_HEIGHT))
#W = int(vs.get(cv2.CAP_PROP_FRAME_WIDTH))

#limitIn = int(H / 2 + H / 5)
#limitOut = int(H / 2 - H / 5)

net = cv2.dnn.readNetFromTensorflow(pbFile, pbtxtFile)

USE_GPU = False

if USE_GPU:
    # set CUDA as the preferable backend and target
    print("[INFO] setting preferable backend and target to CUDA...")
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

global trackers
global trackableObjects
global rects
global liveFPS
# totalIn = 0
# totalOut = 0
camera = None


# function to draw bounding box on the detected object
def drawBoundingBox(frame, box, centroid, color):
    (startX, startY, endX, endY) = box

    # draw a red rectangle around detected objects
    cv2.rectangle(
        frame, (int(startX), int(startY)), (int(endX), int(endY)), color, thickness=2
    )


# return coordinates of the center (centroid) of a bbox
def computeCentroid(box):
    (startX, startY, endX, endY) = box
    return np.array([startX + ((endX - startX) / 2), startY + ((endY - startY) / 2)])


# for every bounding box detected, a trackeable object is created
# to follow his path througn the frame
class TrackableObject:
    def __init__(self, objectID, centroid, zone):
        # store the object ID, then initialize a list of centroids
        # using the current centroid
        self.objectID = objectID
        self.centroids = [centroid]
        self.zone = zone


class CentroidTracker:
    def __init__(self, maxDisappeared=50, maxDistance=80):
        # initialize the next unique object ID along with two ordered
        # dictionaries used to keep track of mapping a given object
        # ID to its centroid and number of consecutive frames it has
        # been marked as "disappeared", respectively
        self.nextObjectID = 0
        self.objects = OrderedDict()
        self.disappeared = OrderedDict()

        # store the number of maximum consecutive frames a given
        # object is allowed to be marked as "disappeared" until we
        # need to deregister the object from tracking
        self.maxDisappeared = maxDisappeared

        # store the maximum distance between centroids to associate
        # an object -- if the distance is larger than this maximum
        # distance we'll start to mark the object as "disappeared"
        self.maxDistance = maxDistance

    def register(self, centroid):
        # when registering an object we use the next available object
        # ID to store the centroid
        self.objects[self.nextObjectID] = centroid
        self.disappeared[self.nextObjectID] = 0
        self.nextObjectID += 1

    def deregister(self, objectID):
        # to deregister an object ID we delete the object ID from
        # both of our respective dictionaries
        del self.objects[objectID]
        del self.disappeared[objectID]

    def update(self, rects):
        # check to see if the list of input bounding box rectangles
        # is empty
        if len(rects) == 0:
            # loop over any existing tracked objects and mark them
            # as disappeared
            for objectID in list(self.disappeared.keys()):
                self.disappeared[objectID] += 1

                # if we have reached a maximum number of consecutive
                # frames where a given object has been marked as
                # missing, deregister it
                if self.disappeared[objectID] > self.maxDisappeared:
                    self.deregister(objectID)

            # return early as there are no centroids or tracking info
            # to update
            return self.objects

        # initialize an array of input centroids for the current frame
        inputCentroids = np.zeros((len(rects), 2), dtype="int")

        # loop over the bounding box rectangles
        for (i, (startX, startY, endX, endY)) in enumerate(rects):
            # use the bounding box coordinates to derive the centroid
            cX = int((startX + endX) / 2.0)
            cY = int((startY + endY) / 2.0)
            inputCentroids[i] = (cX, cY)

        # if we are currently not tracking any objects take the input
        # centroids and register each of them
        if len(self.objects) == 0:
            for i in range(0, len(inputCentroids)):
                self.register(inputCentroids[i])

        # otherwise, are are currently tracking objects so we need to
        # try to match the input centroids to existing object
        # centroids
        else:
            # grab the set of object IDs and corresponding centroids
            objectIDs = list(self.objects.keys())
            objectCentroids = list(self.objects.values())

            # compute the distance between each pair of object
            # centroids and input centroids, respectively -- our
            # goal will be to match an input centroid to an existing
            # object centroid
            D = dist.cdist(np.array(objectCentroids), inputCentroids)

            # in order to perform this matching we must (1) find the
            # smallest value in each row and then (2) sort the row
            # indexes based on their minimum values so that the row
            # with the smallest value as at the *front* of the index
            # list
            rows = D.min(axis=1).argsort()

            # next, we perform a similar process on the columns by
            # finding the smallest value in each column and then
            # sorting using the previously computed row index list
            cols = D.argmin(axis=1)[rows]

            # in order to determine if we need to update, register,
            # or deregister an object we need to keep track of which
            # of the rows and column indexes we have already examined
            usedRows = set()
            usedCols = set()

            # loop over the combination of the (row, column) index
            # tuples
            for (row, col) in zip(rows, cols):
                # if we have already examined either the row or
                # column value before, ignore it
                if row in usedRows or col in usedCols:
                    continue

                # if the distance between centroids is greater than
                # the maximum distance, do not associate the two
                # centroids to the same object
                if D[row, col] > self.maxDistance:
                    continue

                # otherwise, grab the object ID for the current row,
                # set its new centroid, and reset the disappeared
                # counter
                objectID = objectIDs[row]
                self.objects[objectID] = inputCentroids[col]
                self.disappeared[objectID] = 0

                # indicate that we have examined each of the row and
                # column indexes, respectively
                usedRows.add(row)
                usedCols.add(col)

            # compute both the row and column index we have NOT yet
            # examined
            unusedRows = set(range(0, D.shape[0])).difference(usedRows)
            unusedCols = set(range(0, D.shape[1])).difference(usedCols)

            # in the event that the number of object centroids is
            # equal or greater than the number of input centroids
            # we need to check and see if some of these objects have
            # potentially disappeared
            if D.shape[0] >= D.shape[1]:
                # loop over the unused row indexes
                for row in unusedRows:
                    # grab the object ID for the corresponding row
                    # index and increment the disappeared counter
                    objectID = objectIDs[row]
                    self.disappeared[objectID] += 1

                    # check to see if the number of consecutive
                    # frames the object has been marked "disappeared"
                    # for warrants deregistering the object
                    if self.disappeared[objectID] > self.maxDisappeared:
                        self.deregister(objectID)

            # otherwise, if the number of input centroids is greater
            # than the number of existing object centroids we need to
            # register each new input centroid as a trackable object
            else:
                for col in unusedCols:
                    self.register(inputCentroids[col])

        # return the set of trackable objects
        return self.objects


class Stream:

    def __init__(self):
        self.camera_src = inputFile
        self.camera = None
        self.fps = 0
        self.totalIn = 0
        self.totalOut = 0
        self.totalPersonsInside = 0

        self.limitIn = None
        self.limitOut = None
        self.H = None
        self.W = None

    def get_TotalIn(self):
        # global totalIn
        return self.totalIn

    def get_TotalOut(self):
        # global totalOut
        return self.totalOut

    def close(self):
        if self.camera is not None:
            self.camera.release()
            self.camera = None

    def open(self):
        self.camera = cv2.VideoCapture(self.camera_src)  # webcamVideoStream(src=self.camera_src).start()


        # Festlegung der zwei Linien im Bild ab der jemand
        # das Geschäft betritt oder verlässt anhand der Auflösung der Kamera
        self.H = int(self.camera.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.W = int(self.camera.get(cv2.CAP_PROP_FRAME_WIDTH))

        self.limitIn = int(self.H / 2 + self.H / 5)
        self.limitOut = int(self.H / 2 - self.H / 5)

    def status(self):
        return self.camera is not None

    def get_fps(self):
        return self.fps

    def detect(self, frame, detections, trackers, rects):
        # loop over the detections
        for detection in detections[0, 0, :, :]:
            confidence = float(detection[2])

            # if the confidence is above a threshold
            if confidence > minConfidence:
                classID = detection[1]

                # proceed only if the object detected is indeed a human
                if classID == 1:
                    # get coordinates of the bbox
                    left = detection[3] * self.W
                    top = detection[4] * self.H
                    right = detection[5] * self.W
                    bottom = detection[6] * self.H

                    box = [left, top, right, bottom]

                    # construct a dlib rectangle object from the bounding
                    # box coordinates and then start the dlib correlation
                    # tracker
                    tracker = dlib.correlation_tracker()
                    rect = dlib.rectangle(int(left), int(top), int(right), int(bottom))

                    tracker.start_track(frame, rect)

                    trackers.append(tracker)

                    rects.append(box)

                    centroid = computeCentroid(box)

                    drawBoundingBox(frame, box, centroid, color=(0, 0, 255))

                    # cv2.putText(
                    #    frame, status, (0, 115), font, 0.5, (0, 255, 0), 1, cv2.LINE_AA
                    # )

    # object tracking using dlib and centroid tracker
    def track(self, frame, trackers, rects):
        for tracker in trackers:
            status = "Tracking"
            # update the tracker and grab the position of the tracked
            # object
            tracker.update(frame)

            pos = tracker.get_position()

            # unpack the position object
            left = int(pos.left())
            top = int(pos.top())
            right = int(pos.right())
            bottom = int(pos.bottom())

            box = [left, top, right, bottom]

            rects.append(box)

            centroid = computeCentroid(box)

            drawBoundingBox(frame, box, centroid, color=(0, 128, 255))

            # cv2.putText(frame, status, (0, 95), font, 0.5, (0, 255, 0), 1, cv2.LINE_AA)

    # people counting logic based on zone of appearance
    def counting(self, frame, objects, trackableObjects):  # , totalIn, totalOut):

        # loop over the tracked objects
        for (objectID, centroid) in objects.items():

            # check to see if a trackable object exists for the current
            # object ID
            to = trackableObjects.get(objectID, None)

            # if there is no existing trackable object, create one
            if to is None:
                # define starting zone of the trackeable object (IN or OUT)
                if centroid[1] >= self.H / 2:
                    zone = "in"
                else:
                    zone = "out"

                to = TrackableObject(objectID, centroid, zone)

            # otherwise, there is a trackable object so we can utilize it
            # to determine direction
            else:
                if to.zone == "in":
                    if centroid[1] <= self.limitOut:
                        self.totalOut += 1
                        to.zone = "out"
                        # print("OUT : ", totalOut)

                elif to.zone == "out":
                    if centroid[1] >= self.limitIn:
                        self.totalIn += 1
                        to.zone = "in"
                        # print("IN : ", totalIn)

                to.centroids.append(centroid)

            # store the trackable object in our dictionary
            trackableObjects[objectID] = to

            # draw both the ID of the object and the centroid of the
            # object on the output frame
            cv2.circle(frame, (centroid[0], centroid[1]), 4, (0, 0, 255), -1)
            cv2.putText(
                frame,
                "ID : " + str(objectID),
                (centroid[0], centroid[1] + 20),
                font,
                0.6,
                (0, 0, 255),
                1,
                cv2.LINE_AA,
            )

    def generateFrames(self):

        ct = CentroidTracker(maxDisappeared=30, maxDistance=120)

        trackers = []
        trackableObjects = {}

        elapsedFrames = 0

        # start the frames per second throughput estimator
        #self.fps = FPS().start()
        #totalFPS = FPS().start()

        # loop over frames from the video file stream
        while True:
            # read the next frame from the file
            if self.camera is not None:
                (grabbed, frame) = self.camera.read()
                start = time.time()
                self.totalPersonsInside = self.totalIn - self.totalOut

                # frame = imutils.resize(frame, width=300)q

                # if the frame was not grabbed, then we have reached the end
                # of the stream
                if not grabbed:
                    self.camera = cv2.VideoCapture(inputFile)

                    continue

                # list of detected rectangles
                rects = []

                # object detection only every n frames to improve performances
                # strat dlib correlation tracker on detections
                if elapsedFrames % skipFrames == 0:
                    trackers = []
                    status = "Detecting"

                    # Create the blob with a size of (300, 300)
                    blob = cv2.dnn.blobFromImage(
                        frame, size=(inputWidth, inputHeight), swapRB=True, crop=False
                    )

                    # Feed the input blob to the network, perform inference and get the output:
                    # Set the input for the network
                    net.setInput(blob)

                    start = time.time()
                    detections = net.forward()
                    end = time.time()

                    self.detect(frame, detections, trackers, rects)

                # do tracking using detection data on every other frame
                else:
                    self.track(frame, trackers, rects)

                objects = ct.update(rects)
                self.counting(frame, objects, trackableObjects)  # , totalIn, totalOut)

                # draw a two horizontal lines across the frame serving as boundaries
                # one for peoples going IN and an other for OUT
                cv2.line(frame, (0, self.limitIn), (self.W, self.limitIn), (0, 255, 255), 1)
                cv2.line(frame, (0, self.limitOut), (self.W, self.limitOut), (255, 255, 0), 1)

                # construct a tuple of information we will be displaying on the
                # frame
                info = [("Raus", self.totalOut), ("Rein", self.totalIn), ("im Markt", self.totalIn - self.totalOut)]

                # loop over the info tuples and draw them on our frame
                for (i, (k, v)) in enumerate(info):
                    text = "{}: {}".format(k, v)
                    cv2.putText(
                        frame,
                        text,
                        (10, self.H - ((i * 20) + 20)),
                        font,
                        0.6,
                        (0, 255, 255),
                        1,
                        cv2.LINE_AA,
                    )

                # increment the total number of frames processed thus far and
                # then update the FPS counter
                elapsedFrames += 1
                #self.fps.update()

                # show the video beeing processed live
                # cv2.imshow("RPI", frame)

                (flag, encodedImage) = cv2.imencode(".jpg", frame)
                yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + bytearray(encodedImage) + b'\r\n')

                #     # check if the video writer is None
                #     if writer is None:
                #         # initialize our video writer
                #         fourcc = cv2.VideoWriter_fourcc(*"MJPG")
                #         writer = cv2.VideoWriter(outputFile, fourcc, 30, (W, H), True)
                #         print("writing...")

                #     # write the output frame to disk
                #     writer.write(frame)

                # if cv2.waitKey(1) == ord("q"):
                #    break


        # print("[INFO] approx. FPS: {:.2f}".format(totalFPS.fps()))
        # print("OUT : ", totalOut)
        # print("IN : ", totalIn)

        # release the file pointers

        # writer.release()

        # close any open windows
        # cv2.destroyAllWindows()
