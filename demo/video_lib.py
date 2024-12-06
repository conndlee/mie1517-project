

# HELPER FUNCTIONS FOR VIDEO DEMO
# Connor Lee
import cv2 as cv
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision.transforms.functional import gaussian_blur
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

def open_video(n=0):
    capture = cv.VideoCapture(n)
    if not capture.isOpened():  
        print("Error: Could not open video capture.")
        return None
    return capture

class FCCNet(nn.Module):
    def __init__(self):
        super(FCCNet, self).__init__()
        self.name = "FCClassifier"
        self.fc1 = nn.Linear(256 * 6 * 6, 64)
        self.fc2 = nn.Linear(64, 3)

    def forward(self, x):
        x = x.reshape(-1, 256 * 6 * 6)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

class CNNClassifierAlex(nn.Module):
    def __init__(self):
        super(CNNClassifierAlex, self).__init__()
        self.name = "CNNA"
        self.conv1 = nn.Conv2d(256, 160, 5, padding = 2)
        self.fc1 = nn.Linear(160 * 6 * 6, 100)
        self.fc2 = nn.Linear(100, 3)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        self.relu_conv1_out = x
        x = x.view(-1, 160 * 6 * 6) #flatten feature data
        x = F.relu(self.fc1(x))
        self.relu_fc1_out = x
        x = self.fc2(x)
        self.fc2_out = x
        return x
    
class CNNA2(nn.Module):
    def __init__(self):
        super(CNNA2, self).__init__()
        self.name = "CNNAintegrated"
        self.alex  = torchvision.models.alexnet(pretrained=True).features
        for module in self.alex:
            if isinstance(module, nn.ReLU):
                module.inplace = False
        self.classifier = CNNClassifierAlex()

    def forward(self, x):
        x = self.alex(x)
        # print(f"Output from alex: {x.shape}, requires_grad={x.requires_grad}")
        x = F.relu(self.classifier.conv1(x))
        self.relu_conv1_out = x
        x = x.view(-1, 160 * 6 * 6) #flatten feature data
        x = F.relu(self.classifier.fc1(x))
        self.relu_fc1_out = x
        x = self.classifier.fc2(x)
        self.fc2_out = x
        return x
    
def gradient_ascent_max(model, target_neuron, input_shape=(1, 256, 6, 6), steps=100, lr=0.1, regularization=True):

    input_tensor = torch.randn(input_shape, requires_grad=True, device="cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    optimizer = torch.optim.Adam([input_tensor], lr=lr)

    for step in range(steps):
        optimizer.zero_grad()
        output = model(input_tensor)

        target_activation = output[0, target_neuron]

        loss = -target_activation 
        loss.backward()
        optimizer.step()

        if regularization:
            with torch.no_grad():
                input_tensor.clamp_(-1, 1) 
                input_tensor.data = gaussian_blur(input_tensor.data, (3, 3))

        if (step) % 50 == 0:
            print(f"Step {step}/{steps}, Activation: {-loss.item()}")

    return input_tensor.detach()


def exponential_smoothing(data, alpha):
    if not 0 < alpha <= 1:
        raise ValueError("Alpha must be in the range (0, 1].")
    if not data:
        return []

    smoothed = [data[0]] 
    for t in range(1, len(data)):
        smooth_value = alpha * data[t] + (1 - alpha) * smoothed[t - 1]
        smoothed.append(round(smooth_value))  

    return smoothed

def normalize_activation(activation):
    activation -= activation.min() 
    activation /= (activation.max() + 1e-6)  
    activation *= 255  
    return activation.astype(np.uint8)

def create_activation_grid(activation):
    num_filters = activation.shape[1]
    height, width = activation.shape[2], activation.shape[3]

    grid_cols = int(np.ceil(np.sqrt(num_filters)))
    grid_rows = int(np.ceil(num_filters / grid_cols))

    grid_image = np.zeros((grid_rows * height, grid_cols * width), dtype=np.uint8)

    for i in range(num_filters):
        filter_img = normalize_activation(activation[0, i].detach().cpu().numpy())
        row = i // grid_cols
        col = i % grid_cols
        grid_image[row * height:(row + 1) * height, col * width:(col + 1) * width] = filter_img

    return grid_image

def display_activations(activations, name, window_size=(800, 800)):
    for layer_name, activation in activations.items():
        print(f"Visualizing layer: {layer_name}, Shape: {activation.shape}")
        grid_image = create_activation_grid(activation)
        grid_image = cv.resize(grid_image, window_size)
        grid_image = cv.applyColorMap(grid_image, cv.COLORMAP_INFERNO)
        cv.imshow(f"Activations {name} - {layer_name}", grid_image)


def get_activations(model, input_tensor, target_layers):

    activations = {}

    def hook_fn(name):
        def hook(module, input, output):
            activations[name] = output
        return hook

    hooks = []
    for name, layer in model.named_modules():
        if name in target_layers:
            hooks.append(layer.register_forward_hook(hook_fn(name)))

    with torch.no_grad():
        model(input_tensor)

    for hook in hooks:
        hook.remove()

    return activations