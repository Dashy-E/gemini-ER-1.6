import cv2

def get_frame(cap):
    ret, frame = cap.read()
    return ret, frame