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


class Stream:

    def __init__(self, camera_src):
        self.camera_src = camera_src
        self.camera = None
        self.fps = 0
        self.amount_detected = 0
        self.withMask = 0
        self.withoutMask = 0
        self.net = cv2.dnn.readNet('models/face_mask_ssd/face_mask_detection.caffemodel',
                                   'models/face_mask_ssd/face_mask_detection.prototxt')
        # rot und gruen
        self.colors = ((0, 255, 0), (255, 0, 0))
        self.id2class = {0: 'Mask', 1: 'NoMask'}

        self.maskFrames = 0
        self.noMaskFrames = 0

        feature_map_sizes = [[33, 33], [17, 17], [9, 9], [5, 5], [3, 3]]
        anchor_sizes = [[0.04, 0.056], [0.08, 0.11], [0.16, 0.22], [0.32, 0.45], [0.64, 0.72]]
        anchor_ratios = [[1, 0.62, 0.42]] * 5

        # generate anchors
        self.anchors = self.generate_anchors(feature_map_sizes, anchor_sizes, anchor_ratios)

        # for inference , the batch size is 1, the model output shape is [1, N, 4],
        # so we expand dim for anchors to [1, anchor_num, 4]
        self.anchors_exp = np.expand_dims(self.anchors, axis=0)

        self.USE_GPU = True

        if self.USE_GPU:
            # set CUDA as the preferable backend and target
            print("[INFO] setting preferable backend and target to CUDA...")
            self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
            self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

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

    def decode_bbox(self, anchors, raw_outputs, variances=[0.1, 0.1, 0.2, 0.2]):

        anchor_centers_x = (anchors[:, :, 0:1] + anchors[:, :, 2:3]) / 2
        anchor_centers_y = (anchors[:, :, 1:2] + anchors[:, :, 3:]) / 2
        anchors_w = anchors[:, :, 2:3] - anchors[:, :, 0:1]
        anchors_h = anchors[:, :, 3:] - anchors[:, :, 1:2]
        raw_outputs_rescale = raw_outputs * np.array(variances)
        predict_center_x = raw_outputs_rescale[:, :, 0:1] * anchors_w + anchor_centers_x
        predict_center_y = raw_outputs_rescale[:, :, 1:2] * anchors_h + anchor_centers_y
        predict_w = np.exp(raw_outputs_rescale[:, :, 2:3]) * anchors_w
        predict_h = np.exp(raw_outputs_rescale[:, :, 3:]) * anchors_h
        predict_xmin = predict_center_x - predict_w / 2
        predict_ymin = predict_center_y - predict_h / 2
        predict_xmax = predict_center_x + predict_w / 2
        predict_ymax = predict_center_y + predict_h / 2
        predict_bbox = np.concatenate([predict_xmin, predict_ymin, predict_xmax, predict_ymax], axis=-1)
        return predict_bbox

    def generate_anchors(self, feature_map_sizes, anchor_sizes, anchor_ratios, offset=0.5):

        anchor_bboxes = []
        for idx, feature_size in enumerate(feature_map_sizes):
            cx = (np.linspace(0, feature_size[0] - 1, feature_size[0]) + 0.5) / feature_size[0]
            cy = (np.linspace(0, feature_size[1] - 1, feature_size[1]) + 0.5) / feature_size[1]
            cx_grid, cy_grid = np.meshgrid(cx, cy)
            cx_grid_expend = np.expand_dims(cx_grid, axis=-1)
            cy_grid_expend = np.expand_dims(cy_grid, axis=-1)
            center = np.concatenate((cx_grid_expend, cy_grid_expend), axis=-1)

            num_anchors = len(anchor_sizes[idx]) + len(anchor_ratios[idx]) - 1
            center_tiled = np.tile(center, (1, 1, 2 * num_anchors))
            anchor_width_heights = []

            # different scales with the first aspect ratio
            for scale in anchor_sizes[idx]:
                ratio = anchor_ratios[idx][0]  # select the first ratio
                width = scale * np.sqrt(ratio)
                height = scale / np.sqrt(ratio)
                anchor_width_heights.extend([-width / 2.0, -height / 2.0, width / 2.0, height / 2.0])

            # the first scale, with different aspect ratios (except the first one)
            for ratio in anchor_ratios[idx][1:]:
                s1 = anchor_sizes[idx][0]  # select the first scale
                width = s1 * np.sqrt(ratio)
                height = s1 / np.sqrt(ratio)
                anchor_width_heights.extend([-width / 2.0, -height / 2.0, width / 2.0, height / 2.0])

            bbox_coords = center_tiled + np.array(anchor_width_heights)
            bbox_coords_reshape = bbox_coords.reshape((-1, 4))
            anchor_bboxes.append(bbox_coords_reshape)
        anchor_bboxes = np.concatenate(anchor_bboxes, axis=0)
        return anchor_bboxes

    def single_class_non_max_suppression(self, bboxes, confidences, conf_thresh=0.2, iou_thresh=0.5, keep_top_k=-1):

        if len(bboxes) == 0: return []

        conf_keep_idx = np.where(confidences > conf_thresh)[0]

        bboxes = bboxes[conf_keep_idx]
        confidences = confidences[conf_keep_idx]

        pick = []
        xmin = bboxes[:, 0]
        ymin = bboxes[:, 1]
        xmax = bboxes[:, 2]
        ymax = bboxes[:, 3]

        area = (xmax - xmin + 1e-3) * (ymax - ymin + 1e-3)
        idxs = np.argsort(confidences)

        while len(idxs) > 0:
            last = len(idxs) - 1
            i = idxs[last]
            pick.append(i)

            # keep top k
            if keep_top_k != -1:
                if len(pick) >= keep_top_k:
                    break

            overlap_xmin = np.maximum(xmin[i], xmin[idxs[:last]])
            overlap_ymin = np.maximum(ymin[i], ymin[idxs[:last]])
            overlap_xmax = np.minimum(xmax[i], xmax[idxs[:last]])
            overlap_ymax = np.minimum(ymax[i], ymax[idxs[:last]])
            overlap_w = np.maximum(0, overlap_xmax - overlap_xmin)
            overlap_h = np.maximum(0, overlap_ymax - overlap_ymin)
            overlap_area = overlap_w * overlap_h
            overlap_ratio = overlap_area / (area[idxs[:last]] + area[i] - overlap_area)

            need_to_be_deleted_idx = np.concatenate(([last], np.where(overlap_ratio > iou_thresh)[0]))
            idxs = np.delete(idxs, need_to_be_deleted_idx)


        return conf_keep_idx[pick]

    def getOutputsNames(self):
        # Get the names of all the layers in the network
        layersNames = self.net.getLayerNames()
        # Get the names of the output layers, i.e. the layers with unconnected outputs
        return [layersNames[i[0] - 1] for i in self.net.getUnconnectedOutLayers()]

    def inference(self, image, conf_thresh=0.5, iou_thresh=0.4, target_shape=(260, 260)):
        height, width, _ = image.shape
        blob = cv2.dnn.blobFromImage(image, scalefactor=1 / 255.0, size=target_shape)
        self.net.setInput(blob)
        y_bboxes_output, y_cls_output = self.net.forward(self.getOutputsNames())
        # remove the batch dimension, for batch is always 1 for inference.
        y_bboxes = self.decode_bbox(self.anchors_exp, y_bboxes_output)[0]
        y_cls = y_cls_output[0]
        # To speed up, do single class NMS, not multiple classes NMS.
        bbox_max_scores = np.max(y_cls, axis=1)
        bbox_max_score_classes = np.argmax(y_cls, axis=1)

        # keep_idx is the alive bounding box after nms.
        keep_idxs = self.single_class_non_max_suppression(y_bboxes, bbox_max_scores, conf_thresh=conf_thresh,
                                                          iou_thresh=iou_thresh)

        tl = round(0.002 * (height + width) * 0.5) + 1

        self.amount_detected = 0
        self.withMask = 0
        self.withoutMask = 0

        for idx in keep_idxs:
            conf = float(bbox_max_scores[idx])
            class_id = bbox_max_score_classes[idx]
            bbox = y_bboxes[idx]
            # clip the coordinate, avoid the value exceed the image boundary.
            xmin = max(0, int(bbox[0] * width))
            ymin = max(0, int(bbox[1] * height))
            xmax = min(int(bbox[2] * width), width)
            ymax = min(int(bbox[3] * height), height)

            self.amount_detected += 1
            if class_id == 0:
                self.maskFrames += 1
                self.withMask += 1

            else:
                self.noMaskFrames += 1
                self.withoutMask += 1

            label = "{}: {:.2f}%".format(self.id2class[class_id], conf * 100)

            cv2.putText(image, label, (xmin + 2, ymin - 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, self.colors[class_id], 2)
            cv2.rectangle(image, (xmin, ymin), (xmax, ymax), self.colors[class_id], thickness=tl)

        return image

    def generateFrames(self):

        while True:
            if self.camera is not None:
                ret, frame = self.camera.read()
                start = time.time()

                if not ret:
                    break

                # frame = imutils.resize(frame, width=720)
                # frame = cv2.rotate(frame, cv2.ROTATE_180)
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = self.inference(frame, target_shape=(260, 260), conf_thresh=0.5)
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

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
