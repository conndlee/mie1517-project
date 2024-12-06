import torch
import torchvision.models as models
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv
from video_lib import CNNClassifierAlex, CNNA2
from PIL import Image
from ultralytics import YOLO

def compute_cam(model, target_class, input_image, layer_names):
    cams = {}
    activations = {}
    gradients = {}

    def forward_hook(module, input, output, name):
        activations[name] = output

    def backward_hook(module, grad_input, grad_output, name):
        gradients[name] = grad_output[0]

    handles = []
    for name in layer_names:
        layer = dict(model.named_modules())[name]
        handles.append(layer.register_forward_hook(lambda m, i, o, lname=f"{name}": forward_hook(m, i, o, lname)))
        handles.append(layer.register_full_backward_hook(lambda m, gi, go, lname=f"{name}": backward_hook(m, gi, go, lname)))
    

    model.eval()
    input_image = input_image.unsqueeze(0) 
    output = model(input_image)
    print(torch.nn.functional.softmax(output))
    target_score = output[0, target_class]

    model.zero_grad()
    target_score.backward()
    # print(next(iter(model.named_modules())))

    for name in layer_names:
        features = activations[name].squeeze(0)  
        grads = gradients[name].squeeze(0)

        weights = grads.mean(dim=(1, 2))

        cam = torch.zeros(features.shape[1:], dtype=torch.float32, device=features.device)
        for i, w in enumerate(weights):
            cam += w * features[i]

        cam = cam.detach().cpu().numpy()
        cam = np.maximum(cam, 0)
        cam -= cam.min()
        cam /= cam.max() + np.finfo(float).eps
        cam = cv.resize(cam, (input_image.shape[2], input_image.shape[3])) 

        cams[name] = cam

    # Remove hooks
    for handle in handles:
        handle.remove()

    return cams

def visualize_multiple_cams(raw_image, cropped_image, cams, layer_names):
    num_layers = len(layer_names) + 1 +1
    fig, axes = plt.subplots(1, num_layers, figsize=(4 * num_layers, 6))

    axes[0].imshow(raw_image)
    axes[0].set_title("Original Image")
    axes[0].axis("off")
    original_img = cropped_image.permute(1, 2, 0).cpu().numpy()
    original_img = (original_img - original_img.min()) / (original_img.max() - original_img.min())
    original_img = np.uint8(255 * original_img)
    axes[1].imshow(original_img)
    axes[1].set_title("Detected Face")
    axes[1].axis("off")

    for i, layer_name in enumerate(layer_names):
        cam = cams[layer_name]
        cam = cv.applyColorMap(np.uint8(255 * cam), cv.COLORMAP_JET)
        cam = cv.cvtColor(cam, cv.COLOR_RGB2BGR)
        overlay = cv.addWeighted(original_img, 0.5, cam, 0.5, 0)
        axes[i + 2].imshow(overlay)
        axes[i + 2].set_title(layer_name)
        axes[i + 2].axis("off")
    
    plt.tight_layout()
    plt.show()

def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    detection = YOLO('./yolov5su_faces_best.pt')
    print(device)
    classifier = CNNClassifierAlex()
    state = torch.load("./model_CNNA_bs128_lr0.001_epoch10", map_location=device, weights_only=True)
    classifier.load_state_dict(state)
    model = CNNA2()
    # state = torch.load("./model_CNNAintegrated_bs128_lr0.001_epoch14", map_location=device, weights_only=True)
    model.classifier = classifier
    model.eval()
    model.to(device)
    classes = ["glasses", "no glasses", "safety glasses"]
    target_class = 0
    target_layers = [
                    "alex.0",
                    # "alex.1",
                    # "alex.2", 
                    "alex.3", 
                    # "alex.4", 
                    # "alex.5", 
                    "alex.6", 
                    # "alex.7",
                    # "alex.8",
                    # "alex.9",
                    # "alex.10",
                    # "alex.11",
                    "alex.12",
                    "classifier.conv1",
                    ]   

    # print(next(iter(model.named_modules())))
    image_path = f"./imgs/cris_hard{2}.jpg"
    # image_path = f"./imgs/connor_hard{target_class}.png"
    # image = Image.open(image_path).convert("RGB")
    image = cv.imread(image_path)
    image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
    det_results = detection.predict(image, save=False)

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
        im = cv.getRectSubPix(image, (w, h), (x, y))
    
        preprocess = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)), 
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        cropped_image = preprocess(im).to("cuda" if torch.cuda.is_available() else "cpu")

        cams = compute_cam(model, target_class, cropped_image, target_layers)
        visualize_multiple_cams(image, cropped_image, cams, target_layers)

    return 0

if __name__ == "__main__":
    main()