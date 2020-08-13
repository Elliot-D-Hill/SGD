#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May  8 10:17:10 2020

@author: Elliot
"""

import numpy as np
import pandas as pd

class Benchmark:
    
    def __init__(self, sample_evals, target_values, error_list):
        
        self.sample_evals = sample_evals
        self.target_values = target_values
        self.error_list = error_list
    
    def benchmark(self):
        
        target_counter = 0
        num_targets = len(self.target_values)
        evals_at_targets = np.zeros((num_targets))

        for i, err in enumerate(self.error_list):
            
            if target_counter < num_targets:
                if err < self.target_values[target_counter]:
                    evals_at_targets[target_counter] = self.sample_evals[i]
                    target_counter += 1
            else:
                break
            
        return evals_at_targets

    # calculating error in predictions for test set
    def test_error(self, cost_function, X_test, y_test, theta, verbose):
        
        y_pred = cost_function.model(X_test, theta)
        
        if cost_function.method == 'LS': # regression
            err = np.sum(np.square(y_test - y_pred)) / y_test.shape[0]
            if verbose:
                print("Mean absolute error = ", err) 
            majority_err = 'NA'
            minority_err = 'NA'
            
        elif cost_function.method == 'LR': # classification
            y_pred = np.round(y_pred)
            err = 1 - np.mean(y_pred == np.round(y_test))
            
            def class_test_error(class_label):
                data = np.hstack((X_test, y_test)) 
                data = data[data[:, -1] == class_label]
                X_class = data[:,:-1]
                y_class = data[:,-1]
                y_pred_class = cost_function.model(X_class, theta)
                y_pred_class = np.round(y_pred_class)
                class_err = 1 - np.mean(y_pred_class == np.round(y_class))
                return class_err
            
            majority_err = class_test_error(0)
            minority_err = class_test_error(1)
            
            if verbose:
                print("Misclassification rate = ", err)
                print("Majority misclassification rate = ", majority_err)
                print("Minority misclassification rate = ", minority_err)
            
        return y_pred, err, majority_err, minority_err