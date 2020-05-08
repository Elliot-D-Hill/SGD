#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 16 10:41:48 2020

@author: Elliot
"""
# importing dependencies 
import numpy as np 
import SGD as sgd
import CostFunction as cf
import Parameters as pm
import Minibatch as mb
import myPlots as mp
import Dataset as ds
import Results as rs

# gradient descent parameters
parms = pm.Parameters(learning_rate = 0.0001, 
                      batch_size = 64, 
                      tol = 0.01,
                      max_epoch = 5,
                      max_iterations = 2000,
                      decay = 1, 
                      decay_schedule = 1024)

############# choose cost function #############
# choose method from: 'LS' or 'LR'
# choose gradient method from: 'exact' or 'fdm'
cost_function = cf.CostFunction(method = 'LS', grad_method = 'exact')

############# generate data #############
sample_size = 8000
data = ds.create_dataset(cost_function, sample_size, preproc = True)

############# Split data into train and test sets #############
split_factor = 0.9 
X_train, y_train, X_test, y_test = ds.split_train_test(data, split_factor)
  
print("Number of examples in training set = % d"%(X_train.shape[0])) 
print("Number of examples in testing set = % d"%(X_test.shape[0]))

############# Gradient descent #############
numFeatures = data.shape[1] - 1
theta0 = np.zeros((numFeatures, 1)) # initial solution

theta, error_list = sgd.SGD(cost_function, X_train, y_train, theta0, parms)
print("Bias = ", theta[0])
print("Coefficients = ", theta[1:]) 

############# predicting output for test set #############

y_pred, err = rs.test_error(cost_function, X_test, y_test, theta, verbose = True)

############# Plots #############

# plot data
numPoints = 500
mp.plot_data(cost_function, numPoints, data, intercept = False)

# plot gradient descent error
mp.plot_cost(cost_function, error_list)

# plot of model fit
mp.plot_model_fit(cost_function, X_test, y_test, y_pred, theta, intercept = False)

# b = mb.balanced_minibatches('categorical', 4, X_train, y_train, 4)
# print(b)


