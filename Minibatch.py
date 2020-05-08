#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 16 10:30:08 2020

@author: Elliot
"""
import numpy as np

# function to create a list containing mini-batches 
def create_mini_batches(X, y, batch_size): 
    mini_batches = [] 
    data = np.hstack((X, y)) 
    np.random.shuffle(data) 
    n_minibatches = data.shape[0] // batch_size
  
    for i in range(n_minibatches + 1): 
        mini_batch = data[i * batch_size:(i + 1) * batch_size, :]
        X_mini = mini_batch[:, :-1] 
        Y_mini = mini_batch[:, -1].reshape((-1, 1)) 
        mini_batches.append((X_mini, Y_mini))

    if data.shape[0] % batch_size != 0: 
        mini_batch = data[i * batch_size:data.shape[0]]
        X_mini = mini_batch[:, :-1] 
        Y_mini = mini_batch[:, -1].reshape((-1, 1)) 
        mini_batches.append((X_mini, Y_mini)) 
    return mini_batches


def balanced_minibatches(response_type, batch_size, X, y, num_subintervals):
    
    # response_type = y.dtype # FIXME
    mRow = X.shape[0]
    
    # sort data by values of y
    data = np.hstack((X, y))
    data = data[data[:,-1].argsort()]
    print(data)
    
    if response_type == 'categorical':
        unique, counts = np.unique(y, return_counts=True)
        class_counts = dict(zip(unique, counts))
        num_subintervals =  len(unique)
    
    # segregate subintervals into their own arrays
    interval_data = np.empty(num_subintervals, dtype=object)
    idx_start = 0
    idx_end = 0
    
    for i in range(num_subintervals):
        if response_type == 'numeric':
            idx_end = idx_start + (data.shape[0] // num_subintervals)
        elif response_type == 'categorical':
            idx_end = idx_start + class_counts[i]
            
        interval_data[i] = data[idx_start:idx_end]
        idx_start = idx_end
    
    # number of samples per subinterval
    subinterval_size = batch_size // num_subintervals 
    num_batches =  mRow // batch_size # number of mini-batches
    temp_mini_batches = np.empty(num_batches, dtype=object)
    mini_batches = []
    
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


