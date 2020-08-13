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

class Dataset:
    
    # FIXME change from cost_function to problem_type = classification/regression 
    def __init__(self, cost_function, random_seed):
        
        self.random_seed = random_seed
        self.cost_function = cost_function
        
        self.data = []
        self.X_train = []
        self.y_train = []
        self.X_test = []
        self.y_test = []
        

    def create_dataset(self, m_samples, p_features, preproc, balance, noise):
        
        if self.cost_function.method == 'LS': # regression
    
            X, y = datasets.make_regression(n_samples=m_samples, 
                                            n_features=p_features,
                                            n_informative=1, 
                                            noise=noise,
                                            random_state=self.random_seed)
            self.data = np.hstack((X, y.reshape((-1, 1))))
        
            if preproc is True:
                self.data = preprocessing.scale(self.data)
                
            self.data = np.hstack((np.ones((self.data.shape[0], 1)), self.data))
                
        elif self.cost_function.method == 'LR': # classification
    
            X, y = make_classification(n_samples=m_samples, 
                                     n_features=p_features,
                                     n_redundant=0, 
                                     n_informative=1,
                                     weights=[1-balance, balance],
                                     n_clusters_per_class=1,
                                     random_state=np.random.randint(low=0,
                                                                    high=100,
                                                                    size=1)[0])       
            if preproc is True:
                X = preprocessing.scale(X)
                
            X = np.hstack((np.ones((X.shape[0], 1)), X)) 
            y = y[:,np.newaxis]
            self.data = np.hstack((X,y))
            
            return self.data
    
    def split_train_test(self, split_factor):
        
        split = int(split_factor * self.data.shape[0]) 
          
        self.X_train = self.data[:split, :-1] 
        self.y_train = self.data[:split, -1].reshape((-1, 1))
        self.X_test = self.data[split:, :-1] 
        self.y_test = self.data[split:, -1].reshape((-1, 1))
        
        return self.X_train, self.y_train, self.X_test, self.y_test