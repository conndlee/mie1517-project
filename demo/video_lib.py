

# HELPER FUNCTIONS FOR VIDEO DEMO
# Connor Lee
import cv2 as cv

def open_video(n=0):
    capture = cv.VideoCapture(n)
    if not capture.isOpened():  
        print("Error: Could not open video capture.")
        return None
    return capture



