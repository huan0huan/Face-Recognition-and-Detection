# -*- coding: utf-8 -*-
"""
Created on Thu May 23 21:47:52 2019

@author: ASUS
"""

import tensorflow as tf
import cv2
import numpy as np
import os
import random
import sys
from sklearn.model_selection import train_test_split

#建立cnn训练模型
def cnnLayer():
    #第一层
    #卷积核大小（3,3） 通道是3 输出通道32
    W1=weightVariable([3,3,3,32])
    b1=baisVariable([32])

    #卷积
    conv1=tf.nn.relu(conv2d(x,W1)+b1)
    #池化
    pool1=maxPool(conv1)
    #减少过拟合，随机让某些权值不更新
    drop1=dropout(pool1,keep_prob_5)

    #第二层
    W2=weightVariable([3,3,32,64])
    b2=baisVariable([64])
    conv2=tf.nn.relu(conv2d(drop1,W2)+b2)
    pool2=maxPool(conv2)
    drop2=dropout(pool2,keep_prob_5)

    #第三层
    W3=weightVariable([3,3,64,64])
    b3=baisVariable([64])
    conv3=tf.nn.relu(conv2d(drop2,W3)+b3)
    pool3=maxPool(conv3)
    drop3=dropout(pool3,keep_prob_5)

    #全连接层
    Wf=weightVariable([64*64,512])
    bf=baisVariable([512])
    #将特征图展开
    drop3_flat=tf.reshape(drop3,[-1,8*8*64]) #每一张图片的尺寸，是什么时候变为8*8*64的？
    dense=tf.nn.relu(tf.matmul(drop3_flat,Wf)+bf)
    dropf=dropout(dense,keep_prob_75)

    #输出层
    Wout=weightVariable([512,2])
    bout=weightVariable([2])

    out=tf.add(tf.matmul(dropf,Wout),bout)
    return out