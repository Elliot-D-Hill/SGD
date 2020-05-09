#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May  7 21:49:28 2020

@author: Elliot
"""

import numpy as np
from sklearn.datasets import make_classification
from sklearn import datasets
from sklearn import preprocessing

def create_dataset(cost_function, m_samples, p_features, preproc):
    
    if cost_function.method == 'LS': # regression

        X, y = datasets.make_regression(n_samples=m_samples, 
                                        n_features=p_features,
                                        n_informative=1, 
                                        noise=10,
                                        random_state=np.random.randint(low=0,
                                                                   high=100,
                                                                   size=1)[0])
        data = np.hstack((X, y.reshape((-1, 1))))
    
        if preproc is True:
            data = preprocessing.scale(data)
        data = np.hstack((np.ones((data.shape[0], 1)), data))
            
    elif cost_function.method == 'LR': # classification

        X, y = make_classification(n_samples=m_samples, 
                                 n_features=p_features,
                                 n_redundant=0, 
                                 n_informative=1,
                                 n_clusters_per_class=1,
                                 random_state=np.random.randint(low=0,
                                                                high=100,
                                                                size=1)[0])       
        if preproc is True:
            X = preprocessing.scale(X)
            
        X = np.hstack((np.ones((X.shape[0], 1)), X)) 
        y = y[:,np.newaxis]
        data = np.hstack((X,y))
        
    return data

def split_train_test(data, split_factor):
    
    split = int(split_factor * data.shape[0]) 
      
    X_train = data[:split, :-1] 
    y_train = data[:split, -1].reshape((-1, 1))
    X_test = data[split:, :-1] 
    y_test = data[split:, -1].reshape((-1, 1)) 
    
    return X_train, y_train, X_test, y_test