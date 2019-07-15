# -*- coding: utf-8 -*-
"""
Created on Fri May 24 15:58:25 2019

@author: Administrator
"""

'''
TrainCNN
'''

import numpy as np
import tensorflow as tf
#import cv2
#import os
import random
import sys
from sklearn.model_selection import train_test_split
import Functions as Fn
import Blocks

#======常量======
my_faces_path = 'my_faces'
other_faces_path = 'other_faces'
size = 64
epoch = 1#0

#======空列表用于储存数据====== #容器
imgs = []
labs = []

#======读取数据======
Fn.readData(path=my_faces_path, h=size, w=size, imgs=imgs, labs=labs)
Fn.readData(path=other_faces_path, h=size, w=size, imgs=imgs, labs=labs)

#======将图片数据和标签数据转换成Numpy数组======
imgs = np.array(imgs)
labs = np.array(labs)

#======随机划分测试集和训练集======
train_x, test_x, train_y, test_y = train_test_split(imgs, labs, test_size=0.05, 
                                                    random_state = random.randint(0,100)) #[0, 100]随机取一个整数

##======参数 图片数据的总数，图片的 高，宽，通道======
#train_x = train_x.reshape(train_x.shape[0],size,size,3)
#test_x = test_x.reshape(test_x.shape[0],size,size,3)

#======将数据转换成小于1的数====== #归一化
train_x = train_x.astype('float32')/255.0 #np.max(train_x) = 255, np.min(train_x) = 0
test_x = test_x.astype('float32')/255.0 #np.max(test_x) = 255, np.min(test_x) = 0

#======打印训练集和测试集的大小======
print('train size: %s ,test size:%s' % (len(train_x),len(test_x)))

#======分批次 每个批次取100张======
batch_size = 100 #批的大小
num_batch = len(train_x) // batch_size #完整地把训练集训练完，至少需要多少批

#======占位符======
x=tf.placeholder(tf.float32,[None,size,size,3]) #占位符，图片
y_=tf.placeholder(tf.float32,[None,2]) #占位符，标签
keep_prob_5=tf.placeholder(tf.float32)
keep_prob_75=tf.placeholder(tf.float32)

#======网络======
Conv1 = Blocks.Conv(x, 3, 1, 32, keep_prob_5)
Conv2 = Blocks.Conv(Conv1, 3, 1, 64, keep_prob_5)
Conv3 = Blocks.Conv(Conv2, 3, 1, 64, keep_prob_5)
#print(Conv3.shape)
Conv3 = tf.reshape( Conv3, [-1, int(Conv3.shape[1])*int(Conv3.shape[2])*int(Conv3.shape[3])] )
#print(Conv3.shape)
Affine = Blocks.Affine(Conv3, int(Conv3.shape[1]), 512, keep_prob_75)
Output = Blocks.Output(Affine, 512, 2)

#======损失函数======
#print(Output.shape)
#print(y_.shape)
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=Output, labels=y_))

#======优化器======
train_step = tf.train.AdamOptimizer(0.01).minimize(cross_entropy)

#======比较标签是否相等，再求得所有数的平均值======
accuaracy = tf.reduce_mean( tf.cast( tf.equal( tf.argmax(Output,1),tf.argmax(y_,1) ), tf.float32 ) )

#======将loss与accuary 保存以供tensorboard使用======
tf.summary.scalar('loss',cross_entropy)
tf.summary.scalar('accuracy',accuaracy)

##======使用merge_all 可以管理我们的summary 但是容易出错======
#merged_summary_op=tf.summary.merge_all()

#======数据保存器初始化======
saver=tf.train.Saver()

#======训练======
#with tf.Session(config=tf.ConfigProto(log_device_placement=True)) as sess:
#    
#    #初始化所有变量
#    sess.run(tf.global_variables_initializer())
#    summary_writer=tf.summary.FileWriter('./tmp',graph=tf.get_default_graph())
#    index=0
#    
#    for n in range(epoch): #多少个epoch
#        #每次取128（batch_size）张图片
#        for i in range(num_batch): #一个epoch包含多少个批
#            batch_x=train_x[i*batch_size:(i+1)*batch_size] #按照顺序取批，而不是随机取
#            batch_y=train_y[i*batch_size:(i+1)*batch_size]
#
#            #开始训练，同时训练三个变量，返回三个数据
##            _, loss, summary = sess.run([train_step, cross_entropy, merged_summary_op], feed_dict={x:batch_x, y_:batch_y, keep_prob_5: 0.5, keep_prob_75: 0.75})
#            _, loss = sess.run([train_step, cross_entropy], feed_dict={x:batch_x, y_:batch_y, keep_prob_5: 0.5, keep_prob_75: 0.75})
#
##            summary_writer.add_summary(summary,n*num_batch+i)
#            
#            #打印损失 每代
#            print('Step #%d, lossing: %.4f' % (n*num_batch+i,loss))
#            
#            #打印准确率 每100代
#            if (n * num_batch + i) % 100 == 0:
#                #获取测试数据的准确率
#                acc = sess.run(accuaracy,feed_dict={x:test_x, y_:test_y, keep_prob_5: 1.0, keep_prob_75: 1.0})
##                print('======Step #%d , 准确率: %.4f======' % (n*num_batch+i,acc))
#                print( '====== Epoch #%d, 准确率: %.4f ======' % (n, acc) )
#                #当准确率连续十次大于0.99时 保存并退出
#                if acc > 0.99:
#                    index+=1
#                else:
#                    index=0
#                if index == 10:
#                    # model_path=os.path.join(os.getcwd(),'train_faces.model')
#                    saver.save(sess,'./tmp/train_faces.model', global_step=n*num_batch+i)
#                    sys.exit(0)
#                    
#    print('Accuary: ' + str(acc))


#实例化会话
sess = tf.Session()
#初始化所有变量
sess.run(tf.global_variables_initializer())
summary_writer=tf.summary.FileWriter('./tmp',graph=tf.get_default_graph())
index=0

for n in range(epoch): #多少个epoch
    #每次取128（batch_size）张图片
    for i in range(num_batch): #一个epoch包含多少个批
        batch_x=train_x[i*batch_size:(i+1)*batch_size] #按照顺序取批，而不是随机取
        batch_y=train_y[i*batch_size:(i+1)*batch_size]

        #开始训练，同时训练三个变量，返回三个数据
#            _, loss, summary = sess.run([train_step, cross_entropy, merged_summary_op], feed_dict={x:batch_x, y_:batch_y, keep_prob_5: 0.5, keep_prob_75: 0.75})
        _, loss = sess.run([train_step, cross_entropy], feed_dict={x:batch_x, y_:batch_y, keep_prob_5: 0.5, keep_prob_75: 0.75})

#            summary_writer.add_summary(summary,n*num_batch+i)
        
        #打印损失 每代
        print('Step #%d, lossing: %.4f' % (n*num_batch+i,loss))
        
        #打印准确率 每100代
        if (n * num_batch + i) % 100 == 0:
            #获取测试数据的准确率
            acc = sess.run(accuaracy,feed_dict={x:test_x, y_:test_y, keep_prob_5: 1.0, keep_prob_75: 1.0})
#                print('======Step #%d , 准确率: %.4f======' % (n*num_batch+i,acc))
            print( '====== Epoch #%d, 准确率: %.4f ======' % (n, acc) )
            #当准确率连续十次大于0.99时 保存并退出
            if acc > 0.99:
                index+=1
            else:
                index=0
            if index == 10:
                # model_path=os.path.join(os.getcwd(),'train_faces.model')
                saver.save(sess,'./tmp/train_faces.model', global_step=n*num_batch+i)
                sys.exit(0)
                
print('Accuary: ' + str(acc))














