#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May  9 13:45:05 2020

@author: Elliot
"""

import numpy as np 
import Minibatch as mb
import Dataset as ds
import SGD as sgd

def run_experiment(num_problems, 
                   p_features, 
                   sample_size, 
                   num_targets, 
                   num_subintervals, 
                   cost_function,
                   theta0,
                   balance,
                   parms):

    for i in range(num_problems):
        
        ############# choose mini-batch scheme #############

        mini_batcher = mb.Minibatch(parms.batch_size, # mini-batch size
                                    num_subintervals = num_subintervals, # only used for numeric response
                                    response_type = cost_function.method)
        preproc = True
        data = ds.create_dataset(cost_function, sample_size, p_features, preproc, balance)
        
        ############# Split data into train and test sets #############
        
        split_factor = 0.9 
        X_train, y_train, X_test, y_test = ds.split_train_test(data, split_factor)
        
        ############# Gradient descent #############
        
        # Create SGD solver
        solver = sgd.SGD(parms)
        
        theta, error_list, sample_evals = solver.gradient_descent(cost_function, 
                                                                  mini_batcher, 
                                                                  X_train, 
                                                                  y_train, 
                                                                  theta0, 
                                                                  parms)
        
    return error_list, sample_evals

def create_targets(bal_error_list, unbal_error_list, num_targets):
    start = [np.amax(bal_error_list), np.amax(unbal_error_list)]
    start = np.amin(start)
    
    stop = [np.amin(bal_error_list), np.amin(unbal_error_list)]
    stop = np.amax(stop)
    
    step = -abs(start - stop) / num_targets
    
    target_values = np.arange(start=start, stop=stop, step=step)
    
    return target_values