
import cv2 as cv
# import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms

from video_lib import *
from ultralytics import YOLO

classes_det = ["Human face"]
classes_class = ["Glasses", "No Glasses", "Safety Glasses"]
modelpath = './yolov5su_faces_best.pt'
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    ])

if __name__ == "__main__":
    capture = open_video(1) # probably 0
    detection_model = YOLO(modelpath)
    classifier = FCCNet()
    state = torch.load("./model_FCClassifier_bs64_lr0.01_epoch14", map_location=device, weights_only=True)
    classifier.load_state_dict(state)
    classifier.eval()
    classifier.to(device)
    alexnet = torchvision.models.alexnet(pretrained=True).to(device)

    while 1:
        ret, frame = capture.read()
        if not ret:
            print("Error: Unable to read frame.")
            break

        det_results = detection_model.predict(frame, save=False)

        bboxes_toclass = []
        bboxes_todraw = []
        confidences = []
        class_ids = []

        results = det_results[0]
        for box in results.boxes:
            x1, y1, x2, y2 = box.xyxy.tolist()[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            x, y, w, h = box.xywh.tolist()[0]
            x, y, w, h = int(x), int(y), int(w), int(h)
            conf = box.conf.item()
            class_id = int(box.cls.item())

            if conf > 0.5:
                bboxes_toclass.append([x1,y1,x2,y2])
                bboxes_todraw.append([x, y, w, h])
                confidences.append(float(conf))
                class_ids.append(class_id)

        indices = cv.dnn.NMSBoxes(bboxes_toclass, confidences, 0.5, 0.4)
        for i in indices:
            x1, y1, x2, y2 = bboxes_toclass[i]
            x, y, w, h = bboxes_todraw[i]
            label = str(classes_det[class_ids[i]])
            confidence = confidences[i]
            im = cv.getRectSubPix(frame, (w, h), (x, y))
            tf1 = transforms.ToTensor()
            tf2 = transforms.Resize((224, 224))
            im = tf2(tf1(im))
            
            features = alexnet.features(im)
            output = classifier(features)
            prob = torch.nn.functional.softmax(output)
            print(prob)
            print(output)
            _, predicted = torch.max(output.data, 1)
            pred_label = classes_class[predicted.item()]
            cv.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv.putText(frame, f"{pred_label} {prob[0][predicted.item()]:.2f}", (x1, y1 - 10), cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)


        cv.imshow('YOLO Detection + Custom Classification', frame)
        if cv.waitKey(1) & 0xFF == ord('q'):
            break


    capture.release()
    cv.destroyAllWindows()