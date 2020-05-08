#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May  8 10:17:10 2020

@author: Elliot
"""

import numpy as np

# calculating error in predictions for test set
def test_error(cost_function, X_test, y_test, theta, verbose):
    
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