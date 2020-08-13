#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 16 10:29:23 2020

@author: Elliot
"""

import numpy as np

class SGD:
    
    def __init__(self, update_step, parms):
        
        self.update_step = update_step
        self.parms = parms


    def gradient_descent(self, cost_function, mini_batcher, X, y, theta):
        
        iteration = 0
        num_sample_evals = 0
        
        error_list = np.empty((self.parms.max_iterations))
        error_list[:] = np.nan
        sample_evals = np.empty((self.parms.max_iterations))
        sample_evals[:] = np.nan
        
        for itr in range(self.parms.max_epoch):
            
            mini_batches = mini_batcher.create_mini_batches(X, y, self.parms.balanced)
            for mini_batch in mini_batches:
                
                mini_batch_size = len(mini_batch[0])
                
                if mini_batch_size == 0:
                    break
                
                X_mini, y_mini = mini_batch
                
                # calculate gradient
                grad = cost_function.gradient(X_mini, y_mini, theta)
                
                # paramter update step
                theta = self.update_step.update(grad)
                
                # decay schedule reduces step length after number of iterations
                if iteration % self.parms.decay_schedule == 0:
                    self.parms.learning_rate = self.parms.learning_rate * self.parms.decay
                
                # keep track of cost
                err = cost_function.jacobian(X_mini, y_mini, theta)
                error_list[iteration] = err
                
                # keep track of sample evaluations
                num_sample_evals += X_mini.shape[0]
                sample_evals[iteration] = num_sample_evals
                
                gradNorm = np.linalg.norm(grad)
                iteration += 1

                # check stopping criteria
                if gradNorm < self.parms.tol or iteration >= self.parms.max_iterations:
                    break
            if gradNorm < self.parms.tol or iteration >= self.parms.max_iterations:
                break
            
        sample_evals = sample_evals[:iteration]
        error_list = error_list[:iteration]
            
        return theta, error_list, sample_evals