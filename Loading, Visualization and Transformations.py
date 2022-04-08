# -*- coding: utf-8 -*-
"""
Created on Fri Feb  4 18:07:08 2022

@author: gonza
"""

# import packages 

import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from PIL import Image
import cv2


# set path to my working directory

os.chdir('C:/Users/gonza/OneDrive/Desktop/Kernel Methods Data Challenge')

# load the data

def loading():

    Xtr = np.array(pd.read_csv('Xtr.csv',header=None,sep=',',usecols=range(3072)))
    Xte = np.array(pd.read_csv('Xte.csv',header=None,sep=',',usecols=range(3072)))
    Ytr = np.array(pd.read_csv('Ytr.csv',sep=',',usecols=[1])).squeeze()

    Xtr = Xtr[:,0:3072]
    Xte = Xte[:,0:3072]
    Ytr = Ytr
    
    return Xtr, Ytr, Xte


# shape the data into an image to visualize

def shaping(image):

    imR = image[0:1024].reshape(32,32)
    imG = image[1024:2048].reshape(32,32)
    imB = image[2048:3072].reshape(32,32)
    imRGB = np.zeros((32,32,3))
    imRGB[:,:,0] = imR
    imRGB[:,:,1] = imG
    imRGB[:,:,2] = imB
    imRGB = imRGB - imRGB.min()
    imRGB = imRGB / (imRGB.max()-imRGB.min())
    
    return imRGB

# visualization of all the images

def image_vis():
    Xtr, Ytr, Xte = loading()
    for i in range(len(Xtr)):
        im = shaping(Xtr[i,:])
        plt.imshow(im, interpolation='nearest')
        plt.show()
       
# visualization of only one image

def one_image_vis(Xtr,i):
    im = shaping(Xtr[i,:])
    plt.imshow(im, interpolation='nearest')
    plt.show()    

Xtr, Ytr, Xte = loading()


def fromarray2jpg():
    for i in range(1):
        im = (shaping(Xtr[i,:])*255).astype(np.uint8)
        im = Image.fromarray(im)
        im.save(str(i)+'.jpg')
 
def image_vis2():
    Xtr, Ytr, Xte = loading()
    for i in range(len(Xtr)):
        im = shaping(Xtr[i,:])
        im = cv2.GaussianBlur(im,(3,3),0)
        plt.imshow(im, interpolation='nearest')
        plt.show()
  
        
image_vis()
