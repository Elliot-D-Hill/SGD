#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 16 10:58:56 2020

@author: Elliot
"""

class Parameters:
    
    def __init__(self, 
                 grad_method = 'exact',
                 learning_rate = 0.01, 
                 batch_size = 32, 
                 tol = 0.01,
                 max_epoch = 1, 
                 max_iterations = 1000, 
                 decay = 1, 
                 decay_schedule = 50):
        
        self.grad_method = grad_method
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.tol = tol
        self.max_epoch = max_epoch
        self.max_iterations = max_iterations
        self.decay = decay
        self.decay_schedule = decay_schedule
        