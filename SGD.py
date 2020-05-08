#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 16 10:29:23 2020

@author: Elliot
"""

import numpy as np
import Minibatch as mb

def SGD(cost_function, X, y, theta, parms): 

    error_list = []
    iteration = 0
    
    for itr in range(parms.max_epoch):
        
        # create all minibatches for one epoch
        mini_batches = mb.create_mini_batches(X, y, parms.batch_size)
        
        for mini_batch in mini_batches:

            if len(mini_batch[0]) == 0:
                break
            X_mini, y_mini = mini_batch
            
            # calculate gradient
            grad = cost_function.gradient(X_mini, y_mini, theta)
            
            # update step
            theta = theta - parms.learning_rate * grad
            
            # decay schedule reduces step length after number of iterations
            if iteration % parms.decay_schedule == 0:
                parms.learning_rate = parms.learning_rate * parms.decay
            
            # keep track of cost
            err = cost_function.jacobian(X_mini, y_mini, theta)
            error_list.append(err)
                            
            gradNorm = np.linalg.norm(grad)
            iteration += 1
            
            # stopping criteria
            if gradNorm < parms.tol or iteration > parms.max_iterations:
                break
        if gradNorm < parms.tol or iteration > parms.max_iterations:
            break
    return theta, error_list