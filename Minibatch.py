#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 16 10:30:08 2020

@author: Elliot
"""
import numpy as np

class Minibatch:
    
    def __init__(self, batch_size, num_subintervals, response_type):
        self.batch_size = batch_size
        self.num_subintervals = num_subintervals
        self.response_type = response_type
    
    
    def create_mini_batches(self, X, y, balanced):
        
        if np.issubdtype(type(y), np.integer):
            self.response_type = 'classification'
        else:
            self.response_type = 'regression'
            
        
        mRow = X.shape[0]
        mini_batches = []
        data = np.hstack((X, y))
        
        # i.i.d. mini-batches
        if not balanced:
            np.random.shuffle(data) 
            n_minibatches = mRow // self.batch_size
          
            for i in range(n_minibatches + 1):
                
                mini_batch = data[i * self.batch_size:(i + 1) * self.batch_size, :]
                if mini_batch.shape[0] != self.batch_size:
                    break
                X_mini = mini_batch[:, :-1] 
                Y_mini = mini_batch[:, -1].reshape((-1, 1)) 
                mini_batches.append((X_mini, Y_mini))               
        
        # balanced mini-batches
        if balanced:
            # sort data by values of y
            data = data[data[:,-1].argsort()]
            
            if self.response_type == 'classification':
                unique, counts = np.unique(y, return_counts=True)
                class_counts = dict(zip(unique, counts))
                self.num_subintervals =  len(unique)
            
            # segregate subintervals into their own arrays
            interval_data = np.empty(self.num_subintervals, dtype=object)
            idx_start = 0
            idx_end = 0
            
            for i in range(self.num_subintervals):
                
                if self.response_type == 'regression': # least squares
                    idx_end = idx_start + (mRow // self.num_subintervals)
                elif self.response_type == 'classification': # logistic regression
                    idx_end = idx_start + class_counts[i]
                    
                interval_data[i] = data[idx_start:idx_end]
                idx_start = idx_end + 1
            
            # number of samples per subinterval
            subinterval_size = self.batch_size // self.num_subintervals
            num_batches =  mRow // self.batch_size # number of mini-batches
            temp_mini_batches = np.empty(num_batches, dtype=object)            
            
            # create array of balanced mini-batches
            for j in range(num_batches):
                
                    idx = np.random.choice(interval_data[0].shape[0], subinterval_size, replace=False)
                    mini_batch = interval_data[0][idx]
                    
                    for interval_i in range(1, len(interval_data)):
                        
                        idx = np.random.choice(interval_data[interval_i].shape[0], subinterval_size, replace=False)
                        interval_i_batch = interval_data[interval_i][idx]
                        mini_batch = np.vstack((interval_i_batch, mini_batch))
                        
                    temp_mini_batches[j] = mini_batch
                    X_mini = temp_mini_batches[j][:, :-1] 
                    Y_mini = temp_mini_batches[j][:, -1].reshape((-1, 1)) 
                    mini_batches.append((X_mini, Y_mini))
                
        return mini_batches
        


