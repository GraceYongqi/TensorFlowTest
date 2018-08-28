#!/usr/bin/env python
# -*- coding:utf-8 -*-

'''
@File  : test.py
@Author: Grace
@Date  : 2018/8/18
@Desc  : 
'''

# import modules
import numpy
import SharedTools

# global variables
SharedTools.os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
tf = SharedTools.tf
input_data = SharedTools.input_data
mnist = SharedTools.mnist
# class definition

# function definition

# main function
if __name__ == '__main__':
   # input1 = tf.placeholder(tf.float32)
   # input2 = tf.placeholder(tf.float32)
   # output = tf.matmul(input1, input2)
   # with tf.Session() as sess:
   #     print sess.run([output],feed_dict={input1:[[3.,3.]],input2:[[2.],[2.]]})
   #     print sess.run([output], feed_dict={input1: [7.], input2: [2.]})

#download data sets 为了配合后面的softmax学习模型，采用one-hot vector
   #SharedTools里实现，直接调用mnist
#start one session
   sess = tf.InteractiveSession()
#每张图片784个像素点，标签0-9一共10个类别
   x = tf.placeholder("float",shape=[None,784])
   y_ = tf.placeholder("float",shape=[None,10])
#0向量初始化
   W = tf.Variable(tf.zeros([784,10]))
   b = tf.Variable(tf.zeros([10]))
#变量需要在会话里初始化
   sess.run(tf.initialize_all_variables())
#类别预测（预测图片表示的是哪个数字）和损失函数（用于最小化误差）
   #每个像素点都有显示强度，即权重，另外每张图片有偏置量bias
   y = tf.nn.softmax(tf.matmul(x,W)+b)
   #计算交叉熵（tf.reduce_sum将每张图片的交叉熵都加起来了）
   #cross_entropy（成本函数计算得出）越小性能越好
   cross_entropy = -tf.reduce_sum(y_*tf.log(y))
#训练模型
   #以0.01的速率学习，不断优化以降低成本值（梯度下降算法）
   train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)
   #每一次迭代下载50个训练样本，执行一次上述训练过程，用feed_dict将x，y_占位符换成训练集里的数据
   #y_是真实值，学习模型会预测出一个预估值y，在训练时用正确的值去不断优化模型
   for i in range(1000):
       batch = mnist.train.next_batch(50)
       train_step.run(feed_dict={x:batch[0],y_:batch[1]})
#模型完成，评估性能
   #标签都是one-hot向量（只有一个维度是1）标签表示的数字所指向的维度为1（例如 1：【0，1，0，0，0，0，0，0，0】）
   #判断y和y_是否相等返回布尔值，再转换为浮点数，求平均值（argmax给出tensor对象各个维度中数值最大的索引值）
   correct_prediction = tf.equal(tf.argmax(y,1),tf.argmax(y_,1))
   accuracy = tf.reduce_mean(tf.cast(correct_prediction,"float"))
   #上述是计算图，下面.eval填入变量，用测试集数据代替占位符，运行出结果
   print accuracy.eval(feed_dict={x:mnist.test.images,y_:mnist.test.labels})
