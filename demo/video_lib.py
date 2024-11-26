

# HELPER FUNCTIONS FOR VIDEO DEMO
# Connor Lee
import cv2 as cv
import torch.nn as nn
import torch.nn.functional as F

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

