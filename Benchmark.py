#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May  8 10:17:10 2020

@author: Elliot
"""

import numpy as np

class Benchmark:
    
    def __init__(self, sample_evals, 
                 target_values, 
                 error_list):
        
        self.sample_evals = sample_evals
        self.target_values  = target_values
        self.error_list = error_list
    
    def benchmark(self):
        target_sample_eval = []
        target_counter = 0
        for i, err in enumerate(self.error_list):
            if target_counter < len(self.target_values):
                if err < self.target_values[target_counter]:
                    target_sample_eval.append(self.sample_evals[i])
                    target_counter += 1
            else:
                break
        return target_sample_eval

    # calculating error in predictions for test set
    def test_error(self, cost_function, X_test, y_test, theta, verbose):
        
        y_pred = cost_function.model(X_test, theta)
        
        if cost_function.method == 'LS': # regression
            err = np.sum(np.square(y_test - y_pred)) / y_test.shape[0]
            if verbose:
                print("Mean absolute error = ", err) 
            
        elif cost_function.method == 'LR': # classification
            y_pred = np.round(y_pred)
            err = 1 - np.mean(y_pred == np.round(y_test))
            if verbose:
                print("Misclassification rate = ", err)
                
        return y_pred, err