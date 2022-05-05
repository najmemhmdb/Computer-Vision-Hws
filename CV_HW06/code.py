import numpy as np
import cv2
from matplotlib import pyplot as plt

################################### Q3 #########################################
# imgL = cv2.imread('im2.png')
# imgR = cv2.imread('im6.png')
# imgR_gray = cv2.cvtColor(imgR, cv2.COLOR_BGR2GRAY)
# imgL_gray = cv2.cvtColor(imgL, cv2.COLOR_BGR2GRAY)
# patch_size = 19
# region_size = 80
# map = np.zeros(shape=imgR.shape)
# for i in range(imgR.shape[0]):
#     print(i)
#     for j in range(imgR.shape[1]):
#         right_patch = np.zeros(shape=(patch_size, patch_size))
#         startx, endx = max(0, i - (patch_size - 1) / 2), min(i + (patch_size + 1) / 2, imgR.shape[0])
#         starty, endy = max(0, j - (patch_size - 1) / 2), min(j + (patch_size + 1) / 2, imgR.shape[1])
#         ex = startx - i + (patch_size - 1) / 2
#         ey = starty - j + (patch_size - 1) / 2
#         patch = imgR_gray[int(startx):int(endx), int(starty):int(endy)]
#         if ex != 0:
#             patchxend = patch_size
#         else:
#             patchxend = patch.shape[0]
#         if ey != 0:
#             patchyend = patch_size
#         else:
#             patchyend = patch.shape[1]
#         right_patch[int(ex):int(patchxend), int(ey):int(patchyend)] = patch
#         min_distance = np.inf
#         d = 0
#         for col in range(max(0, -1 * region_size + j), min(region_size + j, imgR.shape[1]), 1):
#             left_patch = np.zeros(shape=(patch_size, patch_size))
#             startx, endx = max(0, i - (patch_size - 1) / 2), min(i + (patch_size + 1) / 2, imgR.shape[0])
#             starty, endy = max(0, col - (patch_size - 1) / 2), min(col + (patch_size + 1) / 2, imgR.shape[1])
#             ex = startx - i + (patch_size - 1) / 2
#             ey = starty - col + (patch_size - 1) / 2
#             patch = imgL_gray[int(startx):int(endx), int(starty):int(endy)]
#             if ex != 0:
#                 patchxend = patch_size
#             else:
#                 patchxend = patch.shape[0]
#             if ey != 0:
#                 patchyend = patch_size
#             else:
#                 patchyend = patch.shape[1]
#             left_patch[int(ex):int(patchxend), int(ey):int(patchyend)] = patch
#             left_patch = left_patch.reshape(patch_size ** 2, -1)
            # distance = np.linalg.norm(left_patch - right_patch)
            # if distance < min_distance:
            #     min_distance = distance
            #     d = abs(col - j)
        #
        # map[i, j] = int(d)
# map = map / int(np.max(map))
# plt.imshow(map, cmap='gray')
# plt.show()
################################## Q4 #########################################
imgL = cv2.imread('im2.')
imgR = cv2.imread('im6.jpeg')
imgR_gray = cv2.cvtColor(imgR, cv2.COLOR_BGR2GRAY)
imgL_gray = cv2.cvtColor(imgL, cv2.COLOR_BGR2GRAY)
stereo = cv2.StereoBM_create(numDisparities=5*16,blockSize = 201)
stereo.setPreFilterSize(101)
stereo.setPreFilterCap(1)
stereo.setTextureThreshold(5)
stereo.setUniquenessRatio(0)
stereo.setSpeckleWindowSize(0)
disparity = stereo.compute(imgL_gray, imgR_gray)
plt.imshow(disparity, cmap='gray')
plt.show()
# print(np.min(disparity),np.max(disparity))
#