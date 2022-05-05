import cv2
from time import sleep
import numpy as np
import numpy as np
import cv2
import matplotlib.pyplot as plt
from skimage import data, color, img_as_ubyte
from skimage.feature import canny
from skimage.transform import hough_ellipse
from skimage.draw import ellipse_perimeter

############################ part a ##############################
# img = cv2.imread('sudoku.jpg')
# gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# edges = cv2.Canny(gray, 50, 200)
# lines = cv2.HoughLinesP(edges,150, 90*np.pi/180, 300)
# # img[:,:,:]= 255
# for line in lines:
#   x1, y1, x2, y2 = line[0]
#   cv2.line(img, (x1, y1), (x2, y2), (0, 0, 255),3)
# plt.imshow(img)



############################ part b ##############################
#
#
# vid = cv2.VideoCapture('istockphoto-1179819982-640_adpp_is.mp4')
# width = int(vid.get(3))
# height = int(vid.get(4))
# fps = vid.get(5)
# out = cv2.VideoWriter('output.avi',cv2.VideoWriter_fourcc('M','J','P','G'),fps, (width,height))
# while vid.isOpened():
#   ret,frame = vid.read()
#   if ret == False:
#     break
#
#   gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#   circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1, 500, param1=150, param2=10, minRadius=30, maxRadius=40)
#   # Draw detected circles
#   if circles is not None:
#       circles = np.uint16(np.around(circles))
#       for i in circles[0, :]:
#           # Draw outer circle
#           cv2.circle(frame, (i[0], i[1]), i[2], (0, 0, 255), 2)
#   out.write(frame)
# out.release()

############################ part c ##############################


vid = cv2.VideoCapture(0)
while (True):
    ret, frame = vid.read()
    frame = cv2.resize(frame,(100,100))
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5),cv2.BORDER_DEFAULT)
    edges = cv2.Canny(blur, 150, 200)
    frame = cv2.copyMakeBorder(frame,0,50,0,50,cv2.BORDER_CONSTANT,None,[0,0,0])
    result = hough_ellipse(edges, threshold=10,accuracy=10,min_size=40, max_size=80)
    if len(result) > 0:
        result.sort(order='accumulator')
        best = list(result[-1])
        yc, xc, a, b = [int(round(x)) for x in best[1:5]]
        orientation = best[5]
        cy, cx = ellipse_perimeter(yc, xc, a, b, orientation)
        frame[cy, cx] = (0, 0, 255)
        cv2.imshow('new_frame', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
vid.release()
