
import cv2 as cv
import numpy as np

from video_lib import *
from ultralytics import YOLO

classes = ["Human face"]
modelpath = './checkpointing/yolov5su_faces_best.pt'

if __name__ == "__main__":
    capture = open_video()
    detection_model = YOLO(modelpath)

    while 1:
        ret, frame = capture.read()
        if not ret:
            print("Error: Unable to read frame.")
            break

        height, width, _ = frame.shape
        blob = cv.dnn.blobFromImage(frame, 1/255, (640, 640), (0, 0, 0), True, crop=False)

        det_results = detection_model.predict(blob)

        bounding_boxes = []
        confidences = []
        class_ids = []
        for result in det_results:
            for detection in result:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]

                if confidence > 0.5:
                    center_x   = int(detection[0] * width)
                    center_y   = int(detection[1] * height)
                    box_width  = int(detection[2] * width)
                    box_height = int(detection[3] * height)

                    x = int(center_x - box_width / 2)
                    y = int(center_y - box_height / 2)

                    bounding_boxes.append([x, y, box_width, box_height])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)

        indexes = cv.dnn.NMSBoxes(bounding_boxes, confidences, 0.5, 0.4)
        for i in indexes.flatten():
            x, y, w, h = bounding_boxes[i]
            label = str(classes[class_ids[i]])
            confidence = confidences[i]

            cv.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv.putText(frame, f"{label} {confidence:.2f}", (x, y - 10), cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

        cv.imshow('YOLO Object Detection', frame)
        if cv.waitKey(1) & 0xFF == ord('q'):
            break


    capture.release()
    cv.destroyAllWindows()