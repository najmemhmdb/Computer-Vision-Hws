import cv2
import numpy as np
import matplotlib.pyplot as plt

###################################### part c ###################################
# sift = cv2.SIFT_create()
# trainImg=cv2.imread("book.jpg",0)
# trainImg= cv2.GaussianBlur(trainImg, ksize=(5,5), sigmaX=1.5, sigmaY=1.5)
# keypointsTrain,desTrain = sift.detectAndCompute(trainImg,None)
# # img_with_keypoints = cv2.drawKeypoints(trainImg,keypointsTrain,np.array([]),(0, 0, 255),cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
# # plt.imshow(img_with_keypoints)
# # plt.show()
# vid = cv2.VideoCapture(0)
# endPoints2 = []
# endPoints3 = []
# threshold = 170
# while True:
#     ret, frame =vid.read()
#     gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
#     gray = cv2.GaussianBlur(gray, ksize=(15,15), sigmaX=15, sigmaY=15)
#     keypointsQuery,desQuery = sift.detectAndCompute(gray,None)
#     bf = cv2.BFMatcher(cv2.NORM_L2)
#     matches0 = bf.match(desTrain, desQuery)
#     # msort = [item.distance for item in sorted(matches0, key=lambda x: x.distance)]
#     # print(msort[:5])
#
#     endPoints1 = []
#     for match in matches0:
#         if match.distance > threshold:
#             endPoints1.append(None)
#         else:
#             endPoints1.append(tuple(np.round(keypointsQuery[match.trainIdx].pt).astype(int)))
#     new_frame = frame
#     if len(endPoints1) > 0:
#         if len(endPoints1) != 0 and len(endPoints2) != 0:
#             for i in range(len(endPoints1)):
#                 if endPoints1[i] != None and endPoints2[i] != None:
#                     new_frame = cv2.arrowedLine(frame, endPoints2[i], endPoints1[i], (0, 0, 255), 2)
#         if len(endPoints3) != 0 and len(endPoints2) != 0:
#             for j in range(len(endPoints2)):
#                 if endPoints2[j] != None and endPoints3[j] != None:
#                     new_frame = cv2.arrowedLine(frame, endPoints3[j], endPoints2[j], (0, 255, 0), 2)
#         endPoints3 = endPoints2.copy()
#         endPoints2 = endPoints1
#     else:
#         pass
#     cv2.imshow('ren_frame',new_frame)
#     if cv2.waitKey(10)==ord('q'):
#         break
# vid.release()
# cv2.destroyAllWindows()

###################################### part d ###################################
star = cv2.xfeatures2d.StarDetector_create()
freak = cv2.xfeatures2d.FREAK_create()
trainImg=cv2.imread("book.jpg",0)
trainImg= cv2.GaussianBlur(trainImg, ksize=(5,5), sigmaX=1.5, sigmaY=1.5)
keypoints = star.detect(trainImg,None)
keypointsTrain, desTrain = freak.compute(trainImg,keypoints)
img_with_keypoints = cv2.drawKeypoints(trainImg,keypointsTrain,np.array([]),(0, 0, 255),cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
plt.imshow(img_with_keypoints)
plt.show()
vid = cv2.VideoCapture(0)
endPoints2 = []
endPoints3 = []
threshold = 90
while True:
    ret, frame =vid.read()
    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, ksize=(15,15), sigmaX=15, sigmaY=15)
    kp = star.detect(gray,None)
    keypointsQuery,desQuery = freak.compute(gray,kp)
    new_frame = frame
    if len(keypointsQuery) > 0 :
        bf = cv2.BFMatcher(cv2.NORM_HAMMING)
        matches0 = bf.match(desTrain, desQuery)
        msort = [item.distance for item in sorted(matches0, key=lambda x: x.distance)]
        print(msort[:5])
        matches = []
        for match in matches0:
            if match.distance > threshold:
                matches.append(None)
            else:
                matches.append(match)
        endPoints1 = []
        if(len(matches)>0):
            for match in matches:
                if match == None:
                    endPoints1.append(None)
                else:
                    endPoints1.append(tuple(np.round(keypointsQuery[match.trainIdx].pt).astype(int)))
            if len(endPoints1) != 0 and len(endPoints2) != 0:
                for i in range(len(endPoints1)):
                    if endPoints1[i] != None and endPoints2[i] != None:
                        new_frame = cv2.arrowedLine(frame, endPoints2[i], endPoints1[i], (0, 0, 255), 2)
            if len(endPoints3) != 0 and len(endPoints2) != 0:
                for i in range(len(endPoints2)):
                    if endPoints2[i] != None and endPoints3[i] != None:
                        new_frame = cv2.arrowedLine(frame, endPoints3[i], endPoints2[i], (0, 255, 0), 2)
            endPoints3 = endPoints2.copy()
            endPoints2 = endPoints1
        else:
            pass
    cv2.imshow('result',new_frame)
    if cv2.waitKey(10)==ord('q'):
        break
vid.release()
cv2.destroyAllWindows()