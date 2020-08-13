#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 16 10:41:48 2020

@author: Elliot
"""
# importing dependencies 
import numpy as np
import mySGD as sgd
import CostFunction as cf
import Parameters as pm
import Minibatch as mb
import myPlots as mp
import Dataset as ds
import Experiment as ex
import parameter_update as pu
import Benchmark as bm
import matplotlib.pyplot as plt

############# gradient descent parameters #############

parms = pm.Parameters(learning_rate = 0.001,
                      batch_size = 128, 
                      tol = 0.1,
                      max_epoch = 5,
                      max_iterations = 1000,
                      decay = 1, 
                      decay_schedule = 1,
                      balanced = True)

############# choose cost function #############

cost_function = cf.CostFunction(method = 'LR', # choose from: 'LS' or 'LR'
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

preproc = True
balance = 0.5
noise = 20
random_seed = 1

dataset = ds.Dataset(cost_function, random_seed)
dataset.create_dataset(sample_size, p_features, preproc, balance, noise)

############# Split data into train and test sets #############

split_factor = 0.9 
X_train, y_train, X_test, y_test = dataset.split_train_test(split_factor)

print("Number of examples in training set = % d"%(X_train.shape[0]))
print("Number of examples in testing set = % d"%(X_test.shape[0]))

############# Gradient descent #############

numFeatures = dataset.data.shape[1] - 1
a = 1
b = -1
# random initial solution with elements between -1 and 1
theta0 = (b - a) * np.random.rand(numFeatures, 1) + a

velocity = 0
gamma = 0.99
prev_grad = np.ones((numFeatures))

# choose parameter update method
# update_step = pu.gradient_descent(theta0, parms.learning_rate)
update_step = pu.momentum(theta0, parms.learning_rate, velocity, gamma)
# update_step = pu.predict_correct(theta0, parms.learning_rate, velocity, gamma)
# update_step = pu.scaled_momentum(theta0, parms.learning_rate, velocity, gamma)

# Create SGD solver
solver = sgd.SGD(update_step, parms)

theta, error_list, sample_evals = solver.gradient_descent(cost_function, 
                                                          mini_batcher, 
                                                          X_train, 
                                                          y_train, 
                                                          theta0)
print("Bias = ", theta[0])
print("Coefficients = ", theta[1:]) 

############# Create benchmarker #############

num_targets = 20
target_values = ex.create_targets(error_list, num_targets)

benchmarker = bm.Benchmark(sample_evals, 
                 target_values, 
                 error_list)

############# predicting output for test set #############

y_pred, err, majority_err, minority_err = benchmarker.test_error(cost_function, 
                                                                 X_test, 
                                                                 y_test, 
                                                                 theta, 
                                                                 verbose = True)
############# Plots #############

# plot data
num_points = 200
# plt.figure()
# mp.plot_data(cost_function, num_points, 8, dataset.data)

plt.figure()
# plot gradient descent error
mp.plot_cost(cost_function, error_list)

plt.figure()
# plot of model fit
mp.plot_model_fit(cost_function, X_test, y_test, y_pred, theta)


