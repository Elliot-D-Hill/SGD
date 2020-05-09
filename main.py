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
import Benchmark as bm

############# gradient descent parameters #############
parms = pm.Parameters(learning_rate = 0.001,
                      batch_size = 64, 
                      tol = 0.01,
                      max_epoch = 5,
                      max_iterations = 2000,
                      decay = 1, 
                      decay_schedule = 1024,
                      balanced = True)

############# choose cost function #############

cost_function = cf.CostFunction(method = 'LS', # choose from: 'LS' or 'LR'
                                grad_method = 'exact') # choose: 'exact' or 'fdm'

############# choose mini-batch scheme #############

mini_batcher = mb.Minibatch(parms.batch_size, # mini-batch size
                            num_subintervals = 10, # only used for numeric response
                            response_type = cost_function.method)

############# generate data #############

sample_size = 8000

if cost_function.method == 'LR':
    p_features = 2
elif cost_function.method == 'LS':
    p_features = 1

data = ds.create_dataset(cost_function, sample_size, p_features, preproc = True)

############# Split data into train and test sets #############

split_factor = 0.9 
X_train, y_train, X_test, y_test = ds.split_train_test(data, split_factor)

print("Number of examples in training set = % d"%(X_train.shape[0])) 
print("Number of examples in testing set = % d"%(X_test.shape[0]))

############# Gradient descent #############

numFeatures = data.shape[1] - 1
a = 1
b = -1
# random initial solution with elements between -1 and 1
theta0 = (b - a) * np.random.rand(numFeatures, 1) + a

# Create SGD solver
solver = sgd.SGD(parms)

theta, error_list, sample_evals = solver.gradient_descent(cost_function, 
                                                          mini_batcher, 
                                                          X_train, 
                                                          y_train, 
                                                          theta0, 
                                                          parms)

print("Bias = ", theta[0])
print("Coefficients = ", theta[1:]) 

############# Create benchmarker #############

num_targets = 20
start = np.amax(error_list)
stop = np.amin(error_list)
step = -abs(start - stop) / num_targets
target_values = np.arange(start=start, stop=stop, step=step)

benchmarker = bm.Benchmark(sample_evals, 
                 target_values, 
                 error_list)

############# predicting output for test set #############

y_pred, err = benchmarker.test_error(cost_function, X_test, y_test, theta, verbose = True)

############# Plots #############

# plot data
numPoints = 500
mp.plot_data(cost_function, numPoints, data)

# plot gradient descent error
mp.plot_cost(cost_function, error_list)

# plot of model fit
mp.plot_model_fit(cost_function, X_test, y_test, y_pred, theta)

mp.plot_ecdf(benchmarker.benchmark(), p_features)


