#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May  7 21:49:28 2020

@author: Elliot
"""

import numpy as np
from sklearn.datasets import make_classification
from sklearn import preprocessing

def create_dataset(cost_function, sample_size, preproc):
    if cost_function.method == 'LS': # regression
        # create data 
        mean = np.array([5.0, 6.0]) 
        cov = np.array([[1.0, 0.95], [0.95, 1.2]]) 
        data = np.random.multivariate_normal(mean, cov, sample_size)
    
        if preproc is True:
            data = preprocessing.scale(data)
        
        data = np.hstack((np.ones((data.shape[0], 1)), data))
            
    elif cost_function.method == 'LR': # classification
        # create data
        X, y = make_classification(n_samples=sample_size, n_features=2,
                                 n_redundant=0, n_informative=1,
                                 n_clusters_per_class=1,
                                 random_state=np.random.randint(low=0,high=100,size=1)[0])       
        if preproc is True:
            X = preprocessing.scale(X)
            
        X = np.hstack((np.ones((X.shape[0], 1)), X)) 
        y = y[:,np.newaxis]
        data = np.hstack((X,y));
    return data

def split_train_test(data, split_factor):
    
    split = int(split_factor * data.shape[0]) 
      
    X_train = data[:split, :-1] 
    y_train = data[:split, -1].reshape((-1, 1))
    X_test = data[split:, :-1] 
    y_test = data[split:, -1].reshape((-1, 1)) 
    
    return X_train, y_train, X_test, y_test