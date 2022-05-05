import cv2
from time import sleep
import numpy as np

list_of_frames = []
n = 1
vid = cv2.VideoCapture(0)
while (True):
    ret, frame = vid.read()
    list_of_frames.append(frame / 255)
    if n + 1 < len(list_of_frames):
        new_frame = np.mean(np.array(list_of_frames[len(list_of_frames) - n:]), axis=0)
        cv2.imshow('new_frame', new_frame)
        list_of_frames.clear()
        sleep(0.03)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
vid.release()
