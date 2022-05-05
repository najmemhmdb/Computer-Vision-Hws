import math

import numpy
import numpy as np
import cv2
import matplotlib.pyplot as plt
import random
import os, shutil
import pandas as pd
import numpy as np
import math
from LMFilter import LMFilter
from Schmid import Schmid
from MR import MR8
from itertools import product, chain
from sklearn.cluster import KMeans
from sklearn.metrics.cluster import contingency_matrix
from scipy.spatial.distance import cdist


img_labels = ['Gravel','Grass','Bricks']
imgs = {'Gravel':[],'Grass':[],'Bricks':[]}
# train_inds = random.sample(range(0,7,1),5)
train_inds = [2, 1, 0, 6, 3]
test_inds = [4,5]
# test_inds = set(range(0,7,1)) - set(train_inds)


## read dataset
def load():
    for label in img_labels:
        list = []
        for i in range(1,8,1):
            img_array = cv2.imread('dataset//' + label + "_" + str(i) + ".jpg")
            if img_array is not None:
                img_array = cv2.resize(img_array,(128,128))
            if img_array is None:
                img_array = cv2.resize(cv2.imread('dataset//' + label + "_" + str(i) + ".JPEG"),(128,128))
            list.append(img_array)
        imgs[label] = list
    return

def display_src_imgs():
    p = 0
    for label in img_labels:
        train_imgs = imgs.get(label)
        for ind in train_inds:
            plt.subplot(3,2,p+1)
            p += 1
            plt.imshow(cv2.cvtColor(train_imgs[ind],cv2.COLOR_BGR2RGB))
            plt.title(label + '_' + str(ind))
            plt.axis('off')
    plt.show()
    return


def create_Gabor_Filter_Bank():
    kernels = []
    kernels_parameters = []
    for theta in [90,45,200]:
        for sigma in [1,10]:
            for gamma in [0.5]:
                for lambd in [10]:
                    kernel = np.real(cv2.getGaborKernel([5,5], sigma=sigma, theta=theta*np.pi/180, lambd=lambd,gamma=gamma,ktype=cv2.CV_32F))
                    kernels.append(kernel)
                    kernels_parameters.append('theta'+str(theta)[0:4] + ' sigma' + str(sigma) + ' gamma' + str(gamma) + ' lambd' + str(lambd))
    return kernels,kernels_parameters

def display_Gabor_Filters(kernels,parameters):
    for i in range(len(kernels)):
        plt.subplot(3,2, i+1)
        plt.imshow(kernels[i],cmap='gray')
        plt.axis('off')
        plt.title(parameters[i],fontsize=8)
    plt.show()
    return

# feature extraction by applying filter bank on training images
def apply_Filter_Bank(kernels,kernels_parameters):
    folder = 'dataset/outputs'
    for filename in os.listdir(folder):
        file_path = os.path.join(folder, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print('Failed to delete %s. Reason: %s' % (file_path, e))
    for i, kernel in enumerate(kernels):
        print(i)
        pos = 6
        plt.subplot(4,5,3)
        plt.imshow(kernel,cmap='gray')
        plt.title(kernels_parameters[i])
        plt.axis('off')
        try:
            os.makedirs('dataset//outputs//' + kernels_parameters[i])
        except FileExistsError:
            pass
        for label in img_labels:
            imgs_class = imgs.get(label)
            for j in train_inds:
                try:
                    filterd_img = cv2.filter2D(cv2.cvtColor(imgs_class[j], cv2.COLOR_BGR2GRAY),-1,kernel)
                except:
                    print(label, j)
                plt.subplot(4,5,pos)
                plt.imshow(filterd_img,cmap='gray')
                plt.title(label + '_' + str(j),fontsize=10)
                plt.axis('off')
                pos += 1
                cv2.imwrite('dataset//outputs//' + kernels_parameters[i]+ "//"+label+'_'+str(j)+'.jpg',filterd_img)
        plt.show()
    return


def load_train_filtered_imgs(kernels_parameters):
    filtered_imgs = {'Gravel':[],'Bricks':[],'Grass':[]}
    for filter_path in kernels_parameters:
        for label in img_labels:
            list = filtered_imgs.get(label)
            for i in train_inds:
                img_array = cv2.cvtColor(cv2.imread(
                    'dataset//outputs//' + filter_path + '//' + label + '_' + str(i) + '.jpg'),cv2.COLOR_BGR2GRAY)
                list.append(img_array)
            filtered_imgs[label] = list
    return filtered_imgs

def test(kernels,filtered_imgs):
    for label in img_labels:
        for i in test_inds:
            test_src = imgs.get(label)[i]
            MSE = [0,0,0]
            for k,filter in enumerate(kernels):
                result = cv2.filter2D(cv2.cvtColor(test_src, cv2.COLOR_BGR2GRAY), -1, filter)
                test_features = result.flatten()
                for r,lbl in enumerate(img_labels):
                    for j in range(5):
                        train_imgs = filtered_imgs.get(lbl)
                        train_features = train_imgs[5*k+j].flatten()
                        dist = np.linalg.norm(test_features.mean() - train_features.mean())
                        if dist > MSE[r]:
                            MSE[r] = dist
            print('true label: '  + label + ', predicted label: ' +img_labels[np.argmin(MSE)])


if __name__ == '__main__':
    load()
############################ part a ###############################
    # display_src_imgs()
    # kernels,parameters = create_Gabor_Filter_Bank()
    # display_Gabor_Filters(kernels,parameters)
    # apply_Filter_Bank(kernels,parameters)
############################# part b ##############################
#     filtered_imgs = load_train_filtered_imgs(parameters)
#     test(kernels,filtered_imgs)
############################# part d ##############################

# Leung - Malik Filter Bank
#
    # LM = LMFilter()
    # filter_bank = LM.makeLMfilters()
    # kernels = []
    # for t in range(48):
    #     kernels.append(filter_bank[:,:,t])
    # parameters = ['lmfilter_' + str(i) for i in range(48)]
    # display_Gabor_Filters(kernels)
    # apply_Filter_Bank(kernels,parameters)


# Schmid Filter Bank
#
    # schmid = Schmid()
    # filter_bank = schmid.make_filter_bank()
    # kernels = []
    # for t in range(13):
    #     kernels.append(filter_bank[:,:,t])
    # parameters = ['schmidfilter_' + str(i) for i in range(13)]
    # display_Gabor_Filters(kernels)
    # apply_Filter_Bank(kernels,parameters)


# Maximum Response Filter Bank
#
    # sigmas = [1, 2, 4]
    # n_sigmas = len(sigmas)
    # n_orientations = 6
    # mr8 = MR8()
    # edge, bar, rot = mr8.makeRFSfilters(sigmas=sigmas,n_orientations=n_orientations)
    # n = n_sigmas * n_orientations
    # filterbank = chain(edge, bar, rot)
    # kernels = []
    # for battery in filterbank:
    #     for filter in battery:
    #         kernels.append(filter)
    # print(len(kernels))
    # parameters = ['mr8filter_' + str(i) for i in range(38)]
    # display_Gabor_Filters(kernels)
    # apply_Filter_Bank(kernels,parameters)

############################# part e ##############################
K = range(1,15,1)
p = []

n_filter = 6
kernels,parameters = create_Gabor_Filter_Bank()



# n_filter = 48
# LM = LMFilter()
# filter_bank = LM.makeLMfilters()
# kernels = []
# for t in range(48):
#     kernels.append(filter_bank[:,:,t])

# n_filter = 13
# schmid = Schmid()
# filter_bank = schmid.make_filter_bank()
# kernels = []
# for t in range(13):
#     kernels.append(filter_bank[:,:,t])


# n_filter = 38
# sigmas = [1, 2, 4]
# n_sigmas = len(sigmas)
# n_orientations = 6
# mr8 = MR8()
# edge, bar, rot = mr8.makeRFSfilters(sigmas=sigmas,n_orientations=n_orientations)
# n = n_sigmas * n_orientations
# filterbank = chain(edge, bar, rot)
# kernels = []
# for battery in filterbank:
#     for filter in battery:
#         kernels.append(filter)
#

imgs_features = np.zeros([21,n_filter])
for i,label in enumerate(img_labels):
    for j in range(7):
        f_vector = np.zeros([1,1])
        for filter in kernels:
            f = cv2.filter2D(cv2.cvtColor(imgs.get(label)[j], cv2.COLOR_BGR2GRAY),-1,filter).mean()
            f_vector = np.concatenate((f_vector, f), axis=None)
        f_vector = np.delete(f_vector,0,0)
        imgs_features[i*7+j,:] = f_vector
for k in K:
    kmeans = KMeans(n_clusters=k, random_state=0).fit(imgs_features)
    m = contingency_matrix([0,0,0,0,0,0,0,1,1,1,1,1,1,1,2,2,2,2,2,2,2],kmeans.labels_)
    purity = 0
    p.append(np.sum(np.min(cdist(imgs_features,kmeans.cluster_centers_,metric='euclidean'),axis=1))/21)
    # for i in range(k):
    #     purity += max(m[:,i])
    # purity /= 21
    # p.append(purity)
print(p)
plt.plot(K,p)
plt.xlabel('n_clusters')
# plt.ylabel('purity')
plt.ylabel('MSE')
plt.title('Gabor Filter Bank')
plt.show()

