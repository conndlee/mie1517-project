
import cv2 as cv
# import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms

from collections import deque
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
    detection_model = YOLO(modelpath)
    classifier = CNNClassifierAlex()
    state = torch.load("./model_CNNA_bs128_lr0.001_epoch10", map_location=device, weights_only=True)
    classifier.load_state_dict(state)
    classifier.eval()
    classifier.to(device)
    alexnet = torchvision.models.alexnet(pretrained=True).to(device)
    cap = cv.VideoCapture(1)
    if not cap.isOpened():
        print("Error: Unable to access the camera.")

    preprocess = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    past_predicted = deque(maxlen=5)
    alpha = 0.001

    try:
        while 1:
            ret, frame = cap.read()
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

                input_tensor = preprocess(im).unsqueeze(0)
                features = alexnet.features(input_tensor)
                output = classifier(features)

                target_layers = [
                                #  "0",
                                 "1", 
                                 "2", 
                                #  "3", 
                                #  "4", 
                                 "5", 
                                #  "6", 
                                 "7",
                                #  "8",
                                 "9",
                                #  "10",
                                #  "11",
                                 "12",
                                 ]
                activations = get_activations(alexnet.features, input_tensor, target_layers,)

                name = "alexnet"
                display_activations(activations, name, window_size=(640,620))
                name = "Classifier"
                target_layers = ["conv1"]
                activations = get_activations(classifier, features, target_layers,)
                display_activations(activations, name, window_size=(640,620))

                prob = torch.nn.functional.softmax(output)
                print(output)
                print(prob)
                _, predicted = torch.max(output, 1)
                past_predicted.append(predicted.item())
                stable_pred = exponential_smoothing(past_predicted, alpha)[-1]
                pred_label = classes_class[stable_pred]
                print(past_predicted)
                print(stable_pred)

                cv.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv.putText(frame, f"{pred_label} {prob[0][predicted.item()]:.2f}", (x1, y1 - 10), cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            
            cv.imshow('YOLO Detection + Custom Classification', frame)
            if cv.waitKey(1) & 0xFF == ord('q'):
                break

    finally:
        cap.release()
        cv.destroyAllWindows()