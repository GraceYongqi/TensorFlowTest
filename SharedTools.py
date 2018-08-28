#!/usr/bin/env python
# -*- coding:utf-8 -*-

'''
@File  : SharedTools.py
@Author: Grace
@Date  : 2018/8/28
@Desc  : 
'''

# import modules
import tensorflow as tf
import os
import tensorflow.examples.tutorials.mnist.input_data as input_data


# global variables
#data_path = os.path.join('/tmp/', 'data/')
data_path = os.path.join('/Users/yongqi/PycharmProjects/TensorFlowTest/', 'data/')
mnist = input_data.read_data_sets('MNIST_data',one_hot=True,source_url=data_path)
#mnist = input_data.read_data_sets('MNIST_data',one_hot=True)
# class definition

# function definition

# main function
if __name__ == '__main__':
    pass
