# -*- coding: utf-8 -*-
"""
Created on Fri May 24 19:47:26 2019

@author: Administrator
"""

import numpy as np
import tensorflow as tf
import cv2
#import os
import random
import sys
from sklearn.model_selection import train_test_split
import Functions as Fn
import Blocks

##======常量======
#my_faces_path = 'my_faces'
#other_faces_path = 'other_faces'
#size = 64
#epoch = 10
#
##======空列表用于储存数据====== #容器
#imgs = []
#labs = []
#
##======读取数据======
#Fn.readData(path=my_faces_path, h=size, w=size, imgs=imgs, labs=labs)
#Fn.readData(path=other_faces_path, h=size, w=size, imgs=imgs, labs=labs)
#
##======将图片数据和标签数据转换成Numpy数组======
#imgs = np.array(imgs)
#labs = np.array(labs)
#
##======随机划分测试集和训练集======
#train_x, test_x, train_y, test_y = train_test_split(imgs, labs, test_size=0.05, 
#                                                    random_state = random.randint(0,100)) #[0, 100]随机取一个整数
#
###======参数 图片数据的总数，图片的 高，宽，通道======
##train_x = train_x.reshape(train_x.shape[0],size,size,3)
##test_x = test_x.reshape(test_x.shape[0],size,size,3)
#
##======将数据转换成小于1的数====== #归一化
#train_x = train_x.astype('float32')/255.0 #np.max(train_x) = 255, np.min(train_x) = 0
#test_x = test_x.astype('float32')/255.0 #np.max(test_x) = 255, np.min(test_x) = 0
#
##======打印训练集和测试集的大小======
#print('train size: %s ,test size:%s' % (len(train_x),len(test_x)))
#
##======分批次 每个批次取100张======
#batch_size = 100 #批的大小
#num_batch = len(train_x) // batch_size #完整地把训练集训练完，至少需要多少批
#
##======占位符======
#x=tf.placeholder(tf.float32,[None,size,size,3]) #占位符，图片
#y_=tf.placeholder(tf.float32,[None,2]) #占位符，标签
#keep_prob_5=tf.placeholder(tf.float32)
#keep_prob_75=tf.placeholder(tf.float32)
#
##======网络======
#Conv1 = Blocks.Conv(x, 3, 1, 32, keep_prob_5)
#Conv2 = Blocks.Conv(Conv1, 3, 1, 64, keep_prob_5)
#Conv3 = Blocks.Conv(Conv2, 3, 1, 64, keep_prob_5)
##print(Conv3.shape)
#Conv3 = tf.reshape( Conv3, [-1, int(Conv3.shape[1])*int(Conv3.shape[2])*int(Conv3.shape[3])] )
##print(Conv3.shape)
#Affine = Blocks.Affine(Conv3, int(Conv3.shape[1]), 512, keep_prob_75)
#Output = Blocks.Output(Affine, 512, 2)

#======预测======
Predict = tf.argmax(Output,1)

##======初始化会话，导入模型======
#sess = tf.Session()
#saver = tf.train.Saver()
#saver.restore(sess,tf.train.latest_checkpoint('tmp/'))

#======OpenCV检测人脸======
haar = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
camera = cv2.VideoCapture(0)
flag = 1
while True:
    _,img = camera.read()
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = haar.detectMultiScale(gray_img, 1.3, 5)

    if not len(faces):
        cv2.imshow('img',img)
        key = cv2.waitKey(30)&0xff
        if key == 27:
            sys.exit(0)

    #标记矩形框
    for f_x,f_y,f_w,f_h in faces:
        face = img[f_y:f_y+f_h,f_x:f_x+f_w]
        face = cv2.resize(face,(size,size))
        flag += 1
        res = sess.run(Predict,feed_dict={x:[face/255.0],keep_prob_5:1.0,keep_prob_75:1.0})
        if res[0] == 1:#res[0] == 1:
            print('嗨，我认出你了')
            cv2.imwrite('test_faces/' + str(flag) + '.jpg', face)
            cv2.rectangle(img,(f_x,f_y),(f_x+f_w,f_y+f_h),(0,0,255),3)
        else:
            cv2.rectangle(img,(f_x,f_y),(f_x+f_w,f_y+f_h),(255,0,0),3)
        cv2.imshow('image',img)
        key=cv2.waitKey(30)&0xff
        if key==27:
            sys.exit(0)

sess.close()





