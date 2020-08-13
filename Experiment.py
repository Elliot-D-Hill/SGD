#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May  9 13:45:05 2020

@author: Elliot
"""

import numpy as np 
import Minibatch as mb
import Dataset as ds
import mySGD as gd

def run_problem(p_features, 
                sample_size, 
                num_targets, 
                num_subintervals, 
                cost_function,
                theta0,
                balance,
                noise,
                parms,
                random_seed):
    
    dataset = ds.Dataset(cost_function, random_seed)
        
    ############# choose mini-batch scheme #############

    mini_batcher = mb.Minibatch(parms.batch_size, # mini-batch size
                                num_subintervals = num_subintervals, # only used for numeric response
                                response_type = cost_function.method)
    preproc = True
    dataset.create_dataset(sample_size, 
                            p_features, 
                            preproc, 
                            balance, 
                            noise)
    
    ############# Split data into train and test sets #############
    
    split_factor = 0.9 
    X_train, y_train, X_test, y_test = dataset.split_train_test(split_factor)
    
    ############# Gradient descent #############
    
    # Create SGD solver
    solver = gd.SGD(parms)
    
    theta, error_list, sample_evals = solver.gradient_descent(cost_function, 
                                                              mini_batcher, 
                                                              X_train, 
                                                              y_train, 
                                                              theta0)
    
    return error_list, sample_evals

def create_targets(error_list, num_targets):
    
    start = np.amax(error_list)
    stop = np.amin(error_list)
    target_values = np.linspace(start, stop, num_targets)
    
    return target_values
