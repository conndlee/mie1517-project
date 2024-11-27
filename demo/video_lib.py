

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

class CNNClassifierAlex(nn.Module):
    def __init__(self):
        super(CNNClassifierAlex, self).__init__()
        self.name = "CNNA"
        self.conv1 = nn.Conv2d(256, 160, 5, padding = 2)
        self.fc1 = nn.Linear(160 * 6 * 6, 100)
        self.fc2 = nn.Linear(100, 3)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = x.view(-1, 160 * 6 * 6) #flatten feature data
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


def exponential_smoothing(data, alpha):
    """
    Applies exponential smoothing to a time series of integers.

    Parameters:
    - data (list of int): The time series data to smooth.
    - alpha (float): Smoothing factor (0 < alpha ≤ 1).

    Returns:
    - list of int: Smoothed time series, rounded to integers.
    """
    if not 0 < alpha <= 1:
        raise ValueError("Alpha must be in the range (0, 1].")
    if not data:
        return []

    smoothed = [data[0]]  # Initialize with the first value in the series
    for t in range(1, len(data)):
        smooth_value = alpha * data[t] + (1 - alpha) * smoothed[t - 1]
        smoothed.append(round(smooth_value))  # Round to integer

    return smoothed