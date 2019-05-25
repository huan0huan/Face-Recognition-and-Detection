# -*- coding: utf-8 -*-
"""
Created on Fri May 24 15:00:18 2019

@author: Administrator
"""

'''
Blocks
'''

#import numpy as np
import tensorflow as tf
#import cv2
#import os
#import random
#import sys
#from sklearn.model_selection import train_test_split

def Conv(x, kernel_size, stride, output_channel, keep):
    kernel = tf.Variable( tf.random_normal(mean=0, stddev=0.01, shape=[kernel_size, kernel_size, int(x.shape[3]), output_channel]) )
#    kernel = tf.Variable( (np.sqrt(2./( kernel_size * kernel_size * output_channel ))) * tf.random_normal(mean=0, stddev=1, shape=[kernel_size, kernel_size, int(x.shape[3]), output_channel]) )
    conv = tf.nn.conv2d(x, kernel, [1, stride, stride, 1], padding="SAME")
#    bias = tf.Variable(tf.constant(0.0, shape=[output_channel]))
    bias = tf.Variable(tf.random_normal(shape=[output_channel]))
    relu = tf.nn.relu(tf.nn.bias_add(conv, bias))
    pool = tf.nn.max_pool(relu, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')
    drop = tf.nn.dropout(pool, keep)
    return drop

def Affine(x, input_size, output_size, keep):
    W = tf.Variable( tf.random_normal(mean=0, stddev=0.01, shape=[input_size, output_size]) )
#    b = tf.Variable( tf.constant(0.0, shape=[output_size]) )
    b = tf.Variable( tf.random_normal(shape=[output_size]) )
#    drop3_flat=tf.reshape(drop3,[-1,8*8*64]) #每一张图片的尺寸，经过一些卷积池化之后变为8*8*64
    dense = tf.nn.relu(tf.matmul(x, W) + b)
    drop = tf.nn.dropout(dense, keep)
    return drop

def Output(x, input_size, output_size):
    W = tf.Variable( tf.random_normal(mean=0, stddev=0.01, shape=[input_size, output_size]) )
    b = tf.Variable( tf.random_normal(shape=[output_size]) )
    out = tf.add(tf.matmul(x, W), b)
    return out

    













