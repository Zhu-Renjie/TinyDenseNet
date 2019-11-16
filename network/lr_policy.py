# -*- coding: utf-8 -*- 
# env: python3.6
# author: lm 

import tensorflow as tf 
import numpy as np 


def cyclic_lr(base_lr, max_lr, clr_iterations, step_size = 2000, mode = 'triangular', gamma = 1.,
    scale_fn = None, scale_mode = 'cycle'):
    '''
    https://arxiv.org/abs/1506.01186
    Args:
        base_lr: initial learning rate which is the
            lower boundary in the cycle.
        max_lr:upper boundary in the cycle. Functionally,
            it defines the cycle amplitude (max_lr - base_lr).
            The lr at any cycle is the sum of base_lr
            and some scaling of the amplitude; therefore
            max_lr may not actually be reached depending on
            scaling function.
        step_size: number of training iterations per
            half cycle. Authors suggest setting step_size
            2-8 x training iterations in epoch.
        mode: 'triangular', 'triangular2', 'exp_range'.
            `triangular`:
                A basic triangular cycle w/ no amplitude scaling.
            `triangular2`:
                A basic triangular cycle that scales initial amplitude by half each cycle.
            `exp_range`:
                A cycle that scales initial amplitude by gamma**(cycle iterations) at each
                cycle iteration.
        gamma: constant in 'exp_range' scaling function:
            gamma**(cycle iterations)
        scale_fn: Custom scaling policy defined by a single
            argument lambda function, where
            0 <= scale_fn(x) <= 1 for all x >= 0.
            mode paramater is ignored
        scale_mode: {'cycle', 'iterations'}.
            Defines whether scale_fn is evaluated on
            cycle number or cycle iterations (training
            iterations since start of cycle). Default is 'cycle'.
        
        For more detail, please see paper.
    Return:
        updated learning rate.
    '''
    if scale_fn is None:
        if mode == 'triangular':
            scale_fn = lambda x: 1.0
            scale_mode = 'cycle'
        elif mode == 'triangular2':
            # scale half by each ecpch
            scale_fn = lambda x: 1 / (2. ** (x - 1))
            scale_mode = 'cycle'
        elif mode == 'exp_range':
            scale_fn = lambda x: gamma ** (x)
            scale_mode = 'iterations'
    else:
        # using the specified scale_fn and scale_mode.
        pass 
    # 第几个周期, 注意向下取整
    cycle = tf.cast(tf.floor(1 + clr_iterations / (2 * step_size)), tf.float32)
    x = tf.abs(tf.cast(clr_iterations / step_size, tf.float32) - 2 * cycle + 1)
    if scale_mode == 'cycle':
        print(cycle)
        return base_lr + (max_lr - base_lr) * tf.maximum(0.0, (1 - x) * scale_fn(cycle))
    else:
        return base_lr + (max_lr - base_lr) * tf.maximum(0.0, (1 - x) * scale_fn(clr_iterations)) 
            
    
