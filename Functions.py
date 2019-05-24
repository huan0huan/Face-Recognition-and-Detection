# -*- coding: utf-8 -*-
"""
Created on Fri May 24 15:00:47 2019

@author: Administrator
"""

#import numpy as np
#import tensorflow as tf
import cv2
import os
#import random
#import sys
#from sklearn.model_selection import train_test_split

def getPaddingSize(img): #对于不是正方形的图片，上下左右分别需要补充多少行或者多少列
    h, w, _ = img.shape
    top, bottom, left, right = 0, 0, 0, 0
    longest = max(h, w)
    if w < longest:
        tmp = longest - w
        left = tmp // 2
        right = tmp - left
    elif h < longest:
        tmp = longest - h
        top = tmp // 2
        bottom = tmp - top
    return top, bottom, left, right

def readData(path, h, w, imgs, labs):
    for filename in os.listdir(path): #os.listdir(path)返回指定路径下所有文件和文件夹的名字，并存放于一个列表中。
        if filename.endswith('.jpg'):
            filename = path + '/' + filename #在原有文件名前面添加路径
            img=cv2.imread(filename) #读取图片文件
# ==========补成正方形，再插值成w*h形状的========================================
            top, bottom, left, right = getPaddingSize(img)
            img = cv2.copyMakeBorder(img, top, bottom, left, right, 
                                     cv2.BORDER_CONSTANT, value = [0, 0, 0]) #将图片放大，扩充图片边缘部分
            img = cv2.resize(img, (w, h)) #在cv2.resize(img, (dimension[0], dimension[1]))函数里，dimension[0]是新图片的宽，dimension[1]是新图片的高
# =============================================================================
            imgs.append(img) #添加图片至列表
            if path == 'my_faces': #根据路径添加标签
                labs.append([0,1])
            else:
                labs.append([1,0])




