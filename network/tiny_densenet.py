# -*- coding: utf-8 -*-
# Network for <<DenseNet Models for Tiny ImageNet Classification>>
# author: lm

from __future__ import print_function
import tensorflow as tf 
from network import layers as L


class TinyDenseNet:
    def __init__(self, input_tensor, num_labels, regularizer = None, name = "tiny_densenet"):
        self.inputs = input_tensor # input is nx64x64x3
        self.num_labels = num_labels
        self.name = name
        self.init = tf.truncated_normal_initializer(
                        mean = 0.0, stddev = 0.05)
        self.reg = regularizer
    
    def block(self, prefix, x, kernels = [], is_training = True):
        for i, k in enumerate(kernels):
            name = '{}_{}'.format(prefix, i + 1)
            x = L.conv_bn_relu(name, x, k, (3, 3), (1, 1), regularizer = self.reg)
        return x
    
    def space_to_depth_x2(self, x):
        return tf.space_to_depth(x, block_size = 2)
    
    def network1(self, is_training = True, reuse = False):
        with tf.variable_scope(self.name, reuse = reuse, initializer = self.init):
            # input's shape = (n, 64, 64, 3)
            # block1
            conv1 = self.block('conv1', self.inputs, [32, 64, 128, 256, 512], is_training)      # 64x64
            pool1 = L.pool('max_pool1', conv1, (2, 2), (2, 2))                                  # 32x32
            # block2
            conv2 = self.block('conv2', pool1, [64, 128, 256, 512, 1024], is_training)          # 32x32
            pool2 = L.pool('max_pool2', conv2, (2, 2), (2, 2))                                  # 16x16
            skip1 = self.space_to_depth_x2(pool1)                                               # 16x16
            ccat1 = L.concat('concat1', [skip1, pool2])                                         # 16x16
            # block3
            conv3 = self.block('conv3', ccat1, [32, 128, 256, 512, 1024], is_training)          # 16x16
            pool3 = L.pool('max_pool3', conv3, (2, 2), (2, 2))                                  # 8x8
            skip2 = self.space_to_depth_x2(ccat1)                                               # 8x8
            ccat2 = L.concat('concat2', [skip2, pool3])                                         # 8x8
            # conv classification
            conv4 = L.conv_bn('conv4', ccat2, self.num_labels, (1, 1), 
                              is_training = is_training)                                        # 8x8
            pool4 = L.global_avg_pool('global_avg_pool', conv4)                                 # 1x1
            logits = tf.squeeze(pool4, [1, 2])
            return logits
        
    def network2(self, is_training = True, reuse = False):
        # with tf.variable_scope(self.name, reuse = reuse, initializer = self.init, regularizer = self.reg):
        with tf.variable_scope(self.name, reuse = reuse, initializer = self.init):
            conv1 = self.block('conv1', self.inputs, [32], is_training) # 64x64
            conv2 = self.block('conv2', conv1, [128] * 4, is_training)  # 64x64
            ccat1 = L.concat('ccat1', [conv1, conv2])                   # 64x64
            pool1 = L.pool('max_pool1', ccat1, (2, 2), (2, 2))          # 32x32
            conv3 = self.block('conv3', pool1, [256] * 4, is_training)  # 32x32
            ccat2 = L.concat('ccat2', [pool1, conv3])                   # 32x32
            pool2 = L.pool('max_pool2', ccat2, (2, 2), (2, 2))          # 16x16
            conv4 = self.block('conv4', pool2, [512] * 4, is_training)  # 16x16
            ccat3 = L.concat('ccat3', [pool2, conv4])                   # 16x16
            pool3 = L.pool('max_pool3', ccat3, (2, 2), (2, 2))          # 8x8
            conv5 = L.conv_bn('conv5', pool3, self.num_labels, (1, 1),  # 8x8
                              is_training = is_training, regularizer = self.reg)
            pool4 = L.global_avg_pool('global_avg_pool', conv5)        # 1x1
            logits = tf.squeeze(pool4, [1, 2])
            return logits
        
# FILE END.