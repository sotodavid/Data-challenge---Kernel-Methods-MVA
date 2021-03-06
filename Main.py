# -*- coding: utf-8 -*-
"""Main.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1BQS_hviyAzVVqdA0eFjkM4Qy5i8-NJPy

Load packages
"""

import cvxopt
import sys
from time import time
import numpy as np
import pandas as pd
import matplotlib as plt
import os
from google.colab import drive

"""Mount drive"""

drive.mount('/content/drive/', force_remount=True)
os.chdir('/content/drive/My Drive/Kernel Methods Data Challenge/')

"""Several functions to visualize the images and transform the data can be found in the script "Loading, Visualization and Transformations.py". We used this script as a utility script for our preliminary experiments.

Helper to know remaining time of execution
"""

class Helper:
    
    @staticmethod
    def log_process(title, cursor, finish_cursor, start_time = None):
        percentage = float(cursor + 1)/finish_cursor
        now_time = time()
        time_to_finish = ((now_time - start_time)/percentage) - (now_time - start_time)
        mn, sc = int(time_to_finish//60), int((time_to_finish/60 - time_to_finish//60)*60)
        if start_time:
            sys.stdout.write("\r%s - %.2f%% ----- Temps restant estimé: %d min %d sec -----" %(title, 100*percentage, mn, sc))
            sys.stdout.flush()
        else:
            sys.stdout.write("\r%s - \r%.2f%%" %(title, 100*percentage))
            sys.stdout.flush()

"""We will use SIFT coded from scratch which we have adapted from Svetlana Lazebnik's. It is in the "sift" script.

Image transformations for data augmentation:

SIFT is invariant to scaling and rotation, thus, we have tried other data augmentation techniques:

- Random horizontal flip
- Gaussian blur

We didn't use both at the same time for tractability issues.
"""

from google.colab import drive
drive.mount('/content/drive')

class SIFT:

    def __init__(self, gs = 8, ps = 16, gaussian_thres = 1.0, gaussian_sigma = 0.8, sift_thres = 0.2, \
                 num_angles = 12, num_bins = 5, alpha = 9.0):
        self.num_angles = num_angles
        self.num_bins = num_bins
        self.alpha = alpha
        self.angle_list = np.array(range(num_angles))*2.0*np.pi/num_angles
        self.gs = gs # grid spacing
        self.ps = ps # patch size
        self.gaussian_thres = gaussian_thres
        self.gaussian_sigma = gaussian_sigma
        self.sift_thres = sift_thres
        self.weights = self._get_weights(num_bins)


    def get_params_image(self, image):
        image = image.astype(np.double)
        if image.ndim == 3:
            image = np.mean(image, axis=2)
        H, W = image.shape
        gS = self.gs
        pS = self.ps
        remH = np.mod(H-pS, gS)
        remW = np.mod(W-pS, gS)
        offsetH = remH/2
        offsetW = remW/2
        gridH, gridW = np.meshgrid(range(int(offsetH), H-pS+1, gS), range(int(offsetW), W-pS+1, gS))
        gridH = gridH.flatten()
        gridW = gridW.flatten()
        features = self._calculate_sift_grid(image, gridH, gridW)
        features = self._normalize_sift(features)
        positions = np.vstack((gridH / np.double(H), gridW / np.double(W)))
        return features, positions
    
    def get_X(self, data):
        out = []
        start = time()
        finish = len(data)
        for idx, dt in enumerate(data):
            Helper.log_process('SIFT', idx, finish_cursor=finish, start_time = start)
            out.append(self.get_params_image(np.mean(np.double(dt), axis=2))[0][0])
        return np.array(out)

    def _get_weights(self, num_bins):
        size_unit = np.array(range(self.ps))
        sph, spw = np.meshgrid(size_unit, size_unit)
        sph.resize(sph.size)
        spw.resize(spw.size)
        bincenter = np.array(range(1, num_bins*2, 2)) / 2.0 / num_bins * self.ps - 0.5
        bincenter_h, bincenter_w = np.meshgrid(bincenter, bincenter)
        bincenter_h.resize((bincenter_h.size, 1))
        bincenter_w.resize((bincenter_w.size, 1))
        dist_ph = abs(sph - bincenter_h)
        dist_pw = abs(spw - bincenter_w)
        weights_h = dist_ph / (self.ps / np.double(num_bins))
        weights_w = dist_pw / (self.ps / np.double(num_bins))
        weights_h = (1-weights_h) * (weights_h <= 1)
        weights_w = (1-weights_w) * (weights_w <= 1)
        return weights_h * weights_w

    def _calculate_sift_grid(self, image, gridH, gridW):
        H, W = image.shape
        Npatches = gridH.size
        features = np.zeros((Npatches, self.num_bins * self.num_bins * self.num_angles))
        gaussian_height, gaussian_width = self._get_gauss_filter(self.gaussian_sigma)
        IH = self._convolution2D(image, gaussian_height)
        IW = self._convolution2D(image, gaussian_width)
        Imag = np.sqrt(IH**2 + IW**2)
        Itheta = np.arctan2(IH,IW)
        Iorient = np.zeros((self.num_angles, H, W))
        for i in range(self.num_angles):
            Iorient[i] = Imag * np.maximum(np.cos(Itheta - self.angle_list[i])**self.alpha, 0)
        for i in range(Npatches):
            currFeature = np.zeros((self.num_angles, self.num_bins**2))
            for j in range(self.num_angles):
                currFeature[j] = np.dot(self.weights,\
                        Iorient[j,gridH[i]:gridH[i]+self.ps, gridW[i]:gridW[i]+self.ps].flatten())
            features[i] = currFeature.flatten()
        return features

    def _normalize_sift(self, features):
        siftlen = np.sqrt(np.sum(features**2, axis=1))
        hcontrast = (siftlen >= self.gaussian_thres)
        siftlen[siftlen < self.gaussian_thres] = self.gaussian_thres
        features /= siftlen.reshape((siftlen.size, 1))
        features[features>self.sift_thres] = self.sift_thres
        features[hcontrast] /= np.sqrt(np.sum(features[hcontrast]**2, axis=1)).\
                reshape((features[hcontrast].shape[0], 1))
        return features


    def _get_gauss_filter(self, sigma):
        gaussian_filter_amp = np.int(2*np.ceil(sigma))
        gaussian_filter = np.array(range(-gaussian_filter_amp, gaussian_filter_amp+1))**2
        gaussian_filter = gaussian_filter[:, np.newaxis] + gaussian_filter
        gaussian_filter = np.exp(- gaussian_filter / (2.0 * sigma**2))
        gaussian_filter /= np.sum(gaussian_filter)
        gaussian_height, gaussian_width = np.gradient(gaussian_filter)
        gaussian_height *= 2.0/np.sum(np.abs(gaussian_height))
        gaussian_width  *= 2.0/np.sum(np.abs(gaussian_width))
        return gaussian_height, gaussian_width
    
    def _convolution2D(self, image, kernel):
        imRows, imCols = image.shape
        kRows, kCols = kernel.shape

        y = np.zeros((imRows,imCols))

        kcenterX = kCols//2
        kcenterY = kRows//2

        for i in range(imRows):
            for j in range(imCols):
                for m in range(kRows):
                    mm = kRows - 1 - m
                    for n in range(kCols):
                        nn = kCols - 1 - n

                        ii = i + (m - kcenterY)
                        jj = j + (n - kcenterX)

                        if ii >= 0 and ii < imRows and jj >= 0 and jj < imCols :
                            y[i][j] += image[ii][jj] * kernel[mm][nn]

        return y

class ImageTransformation:
    
    def flip_image_horizontal(image):
        # Takes an image as input and outputs the same image with a horizontal flip
        result = image.copy()
        for channel in range(3):
            aux = image[:, :, channel]
            for column in range(len(aux)):
                result[:, column, channel] = aux[:, len(aux) - column - 1]
        return result

# I will try to add Gaussian Blur
import cv2
class ImageTransformation:
    
    def gaussian_blurr(image):
        # Takes an image as input and outputs the same image with a blurr
        result = cv2.GaussianBlur(image,(5,5),cv2.BORDER_DEFAULT)      
        return result

"""Load datasets"""

# Training set
X_df = pd.read_csv('/content/drive/MyDrive/Kernel Methods Data Challenge/Xtr.csv', header=None)
y_df = pd.read_csv('/content/drive/MyDrive/Kernel Methods Data Challenge/Ytr.csv')
X_df = X_df.loc[:,:3071]

# Test set
X_test = pd.read_csv('/content/drive/MyDrive/Data challenge 1/Xte.csv', header=None)
X_test = X_test.loc[:,:3071]

X = X_df.values
y = y_df.Prediction

X_test = X_test.values

"""Train and test image processing into 32x32x3 shape"""

red, green, blue = np.hsplit(X, 3)
data = np.array([np.dstack((red[i], blue[i], green[i])).reshape(32, 32, 3) for i in range(len(X))])

red, green, blue = np.hsplit(X_test, 3)
data_test = np.array([np.dstack((red[i], blue[i], green[i])).reshape(32, 32, 3) for i in range(len(X_test))])

"""Data augmentation with Gaussian blur or random horizontal flip."""

# Flipping image from the train set
start = time()
finish = len(data)
augmented_train = []

for row in range(0, finish):
    if row % 50 == 0 or row == finish-1:
        Helper.log_process('Performing Gaussian Blur on the images...', row, finish_cursor=finish, start_time = start)
    augmented_train.append(data[row])
    augmented_train.append(ImageTransformation.gaussian_blurr(data[row]))
    # augmented_train.append(ImageTransformation.flip_image_horizontal(data[row]))
augmented_train=np.array(augmented_train)
    
start = time()
augmented_labels = []
for row in range(len(data)):
    lab = y[row]
    augmented_labels.append(lab)
    augmented_labels.append(lab)   
augmented_labels = np.array(augmented_labels)

"""We set SIFT parameters, the choice of them has been done based on existing information and empirically."""

params = { 'gs': 6,
           'chi2_gamma': .6,
           'C': 10.,
           'ps': 31,
           'sift_thres': .3,
           'gaussian_thres': .7,
           'gaussian_sigma': .4,
           'num_angles': 12,
           'num_bins': 5,
           'alpha': 9.0 }

extractor = SIFT(gs=params['gs'], 
                 ps=params['ps'], 
                 sift_thres=params['sift_thres'], 
                 gaussian_sigma=params['gaussian_sigma'], 
                 gaussian_thres=params['gaussian_thres'],
                 num_angles=params['num_angles'],
                 num_bins=params['num_bins'],
                 alpha=params['alpha'])

"""With data augmentation we end up with 10000 vectors of dimension $n$, the number of SIFT features, for training."""

target = augmented_labels
train = extractor.get_X(augmented_train)
test = extractor.get_X(data_test)

"""Without data augmentation we have 5000 vectors of dimension $n$, the number of SIFT features, for training."""

#train = extractor.get_X(data)
#target = y
#test = extractor.get_X(data_test)

"""CRUCIAL: center and scale feaures. Otherwise 25% will be the maximum reachable."""

# Center,reduce AFTER having extracted the features

from scale import scale  

USE_SCALE_AFTER_FEATURES = True

if USE_SCALE_AFTER_FEATURES:
    print("Post-processing the data, calling scale() on the transformed data: centering so mu = 0, scaling so std = 1 ...")
    train = scale(train, copy=True)
    test = scale(test, copy=True)
else:
    print("No post-processing the data, not calling scale() on the transformed data ...")

# Parameters for SVC.

C = 7.5  # Our best score was obtained with 7.5 and data aug from horizontal flip
kernel = 'rbf'  # The best so is the one we use
gamma = 0.001  # Seems good by GridSearch, could have used 1/n where n is the number of samples
coef0 = 0.0
degree = 5  # Irrelevant if not for poly
cache_size = 250  

# Set of parameters for the SVC classifier
svm_parameters = {'C': C,
                  'kernel': kernel,
                  'gamma': gamma,
                  'degree': degree,
                  'coef0': coef0,
                  'cache_size': cache_size}
print("Using these parameters for the SVM classifier:", svm_parameters)

# Train model on known data

from svm import mySVC

svm_model = mySVC(**svm_parameters)
print("Training SVM on all training data ...")
target = np.array(target) 
svm_model.fit(train, target)

# Verify results on known data (gives you a hint on whether you are overfitting)

CHECK_SCORE_TRAIN_DATA = True
if CHECK_SCORE_TRAIN_DATA:
    train_score = svm_model.score(train, target)
    print("Checking the score on the train data : {:.2%} ...".format(train_score))

# Prediction on non labelled data
print("Prediction on test data ...")
prediction = svm_model.predict(test)

# Saving the predictions

outname = 'Submission_SVM_Kernel%s_C=%s__Soto_Reinoso.csv' % (kernel, str(C))
print("Saving the predictions to the CSV file '%s' ..." % outname)

# Saving to 'outname', and to 'Yte.csv' 
for on in [outname, 'Yte.csv']:
    np.savetxt(on,
               np.c_[range(1, len(test) + 1), prediction],
               delimiter=',',
               comments='',
               header='Id,Prediction',
               fmt='%d')