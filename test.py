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
import tensorflow as tf
import os
import tensorflow.examples.tutorials.mnist.input_data

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# global variables

# class definition

# function definition

# main function
if __name__ == '__main__':
    input1 = tf.placeholder(tf.float32)
    input2 = tf.placeholder(tf.float32)
    output = tf.matmul(input1, input2)

    with tf.Session() as sess:
        print sess.run([output],feed_dict={input1:[[3.,3.]],input2:[[2.],[2.]]})
   #     print sess.run([output], feed_dict={input1: [7.], input2: [2.]})
