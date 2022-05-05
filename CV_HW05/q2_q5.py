import numpy as np
import cv2 as cv

###################################### part b ###################################
# cap = cv.VideoCapture(0)
# fourcc = cv.VideoWriter_fourcc('X', 'V', 'I', 'D')
# videoWriter = cv.VideoWriter('output.avi', fourcc, 30.0, (640, 480))
#
# feature_params = dict( maxCorners = 4,
#                        qualityLevel = 0.3,
#                        minDistance = 7,
#                        blockSize = 7 )
# # Parameters for lucas kanade optical flow
# lk_params = dict( winSize  = (15, 15),
#                   maxLevel = 2,
#                   criteria = (cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 0.03))
# ret, old_frame = cap.read()
# old_gray = cv.cvtColor(old_frame, cv.COLOR_BGR2GRAY)
# old_gray = cv.GaussianBlur(old_gray, ksize=(15, 15), sigmaX=15, sigmaY=15)
# p1 = cv.goodFeaturesToTrack(old_gray, mask = None, **feature_params)
# endpoints1 = []
# endpoints0 = []
# while True:
#     ret, frame = cap.read()
#     frame_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
#     frame_gray = cv.GaussianBlur(frame_gray, ksize=(15, 15), sigmaX=15, sigmaY=15)
#     # calculate optical flow
#     p1, state, err = cv.calcOpticalFlowPyrLK(old_gray, frame_gray, p1, None, **lk_params)
#     old_gray = frame_gray.copy()
#     endpoints2 = []
#     if len(p1) > 0:
#         for i,s in enumerate(state):
#             if s == 1:
#                 endpoints2.append(p1[i])
#             else:
#                 endpoints2.append(None)
#     if len(endpoints2) > 0 and len(endpoints1) > 0:
#         for old,new in zip(endpoints1,endpoints2):
#             if old is not None and new is not None:
#                 old = old[0].astype(int)
#                 new = new[0].astype(int)
#                 frame = cv.arrowedLine(frame, old, new, (255,0,0), 2)
#     if len(endpoints1) > 0 and len(endpoints0) > 0:
#         for old, new in zip(endpoints0, endpoints1):
#             if old is not None and new is not None:
#                 old = old[0].astype(int)
#                 new = new[0].astype(int)
#                 frame = cv.arrowedLine(frame, old, new, (0, 255, 0), 2)
#     endpoints0 = endpoints1.copy()
#     endpoints1 = endpoints2.copy()
#     cv.imshow('result', frame)
#     videoWriter.write(frame)
#     if cv.waitKey(10) == ord('q'):
#         break
# cap.release()
# videoWriter.release()
# cv.destroyAllWindows()

###################################### part e ###################################
# import cv2
# import numpy as np
#
#
# def draw_flow(img, flow, step, x, y,color):
#     h, w = img.shape[:2]
#     if len(x) == 0:
#         y, x = np.mgrid[step / 2:h:step, step / 2:w:step].reshape(2, -1).astype(int)
#     y = np.clip(y, 0, h - 1)
#     x = np.clip(x, 0, w - 1)
#     fx, fy = flow[y, x].T
#     lines = np.vstack([x, y, x + fx, y + fy]).T.reshape(-1, 2, 2)
#     lines = np.round(lines).astype(int)
#     vis = img
#     for (x1, y1), (x2, y2) in lines:
#         if x1 - x2 < 2 and y1 - y2 < 2:
#             continue
#         cv2.arrowedLine(vis, (x1, y1), (x2, y2), color, 2)
#     return vis, lines[:, 1, 0], lines[:, 1, 1]
#
#
# cap = cv2.VideoCapture(0)
# fourcc = cv2.VideoWriter_fourcc('X', 'V', 'I', 'D')
# videoWriter = cv2.VideoWriter('output2.avi', fourcc, 5.0, (640, 480))
# _, prev = cap.read()
# prev_gray = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)
# x, y = [], []
# prev2 = []
# while True:
#     _, frame = cap.read()
#     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#     gray = cv2.GaussianBlur(gray, (9, 9), 3, 3)
#     vis = frame
#     if len(prev2) > 0:
#         flow = cv2.calcOpticalFlowFarneback(prev2, prev_gray, None, 0.5, 3, 20, 3, 5, 1.2, 0)
#         vis, x, y = draw_flow(frame, flow, 100, x, y, (0, 0, 255))
#     flow = cv2.calcOpticalFlowFarneback(prev_gray, gray, None, 0.5, 3, 20, 3, 5, 1.2, 0)
#     vis, x, y = draw_flow(vis, flow, 100, x, y, (0, 255, 0))
#     cv2.imshow('Optical flow', vis)
#     videoWriter.write(vis)
#     prev2 = prev_gray.copy()
#     prev_gray = gray.copy()
#     if cv2.waitKey(20) == ord('q'):
#         break
# cap.release()
# videoWriter.release()
# cv2.destroyAllWindows()
###################################### part e ###################################

# Importing libraries
import cv2
import numpy as np

capture = cv2.VideoCapture(0)
fourcc = cv2.VideoWriter_fourcc('X', 'V', 'I', 'D')
videoWriter = cv2.VideoWriter('output2.avi', fourcc, 9.0, (640, 480))
_, frame1 = capture.read()
prvs = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
hsv_mask = np.zeros_like(frame1)
hsv_mask[..., 1] = 255
while True:
    _, frame2 = capture.read()
    next = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
    flow = cv2.calcOpticalFlowFarneback(prvs, next, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    hsv_mask[..., 0] = ang * 180 / np.pi / 2
    hsv_mask[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
    rgb_representation = cv2.cvtColor(hsv_mask, cv2.COLOR_HSV2BGR)
    cv2.imshow('frame2', rgb_representation)
    videoWriter.write(rgb_representation)
    kk = cv2.waitKey(20) & 0xff
    if kk == ord('q'):
        break
    prvs = next
capture.release()
videoWriter.release()
cv2.destroyAllWindows()
