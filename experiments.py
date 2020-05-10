#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May  8 15:26:38 2020

@author: Elliot
"""
# importing dependencies 
import numpy as np 
import pandas as pd
import CostFunction as cf
import Parameters as pm
import Experiment as ex
import Benchmark as bm


# experimental parameters
dimensions = [2, 4, 8, 16, 32, 64]
class_imbalance = np.around(np.arange(start=0.1, stop=0.6, step=0.1), decimals=1)
class_imbalance = np.insert(class_imbalance, 0, 0.01)
mini_balance = class_imbalance
subintervals = [4, 8, 16, 32, 64, 128]   


############# gradient descent parameters #############
parms = pm.Parameters(learning_rate = 0.001,
                      batch_size = 64, 
                      tol = 0.01,
                      max_epoch = 5,
                      max_iterations = 100,
                      decay = 1, 
                      decay_schedule = 1024,
                      balanced = True)

############# choose cost function #############

cost_function = cf.CostFunction(method = 'LS', # choose from: 'LS' or 'LR'
                                grad_method = 'exact') # choose: 'exact' or 'fdm'

############# generate experimental data #############

num_problems = 30
sample_size = 4000
num_targets = 10
# num_subintervals = 64
dim = 32
balance = 0.5

balanced_dim_df = pd.DataFrame()
unbalanced_dim_df = pd.DataFrame()

balanced_df = pd.DataFrame()
unbalanced_df = pd.DataFrame()

exp_var = subintervals
for i, num_subintervals in enumerate(exp_var):
    
    # random initial solution
    a = 1
    b = -1
    # random initial solution with elements between -1 and 1
    theta0 = (b - a) * np.random.rand(dim+1, 1) + a
    
    # allow more iterations in higher dimensions
    parms.max_iterations = parms.max_iterations * dim

    bal_error_list, bal_sample_evals = ex.run_experiment(num_problems, 
                                   dim, 
                                   sample_size, 
                                   num_targets,
                                   num_subintervals,
                                   cost_function,
                                   theta0,
                                   balance,
                                   parms)
    
    parms.balanced = False
    
    unbal_error_list, unbal_sample_evals = ex.run_experiment(num_problems, 
                                    dim, 
                                    sample_size, 
                                    num_targets,
                                    num_subintervals,
                                    cost_function,
                                    theta0,
                                    balance,
                                    parms)
    
    
    ############# Create benchmarkers #############

    target_values = ex.create_targets(bal_error_list, unbal_error_list, num_targets)
    
    bal_benchmarker = bm.Benchmark(bal_sample_evals, 
                     target_values, 
                     bal_error_list)
    
    unbal_benchmarker = bm.Benchmark(unbal_sample_evals, 
                     target_values, 
                     unbal_error_list)
    
    column_name = "problem_" + str(i)
    
    # balanced
    evals_at_targets = bal_benchmarker.benchmark()
    balanced_df[column_name] = evals_at_targets
    # unbalanced
    evals_at_targets = unbal_benchmarker.benchmark()
    unbalanced_df[column_name] = evals_at_targets

    
    balanced_mean = np.mean(balanced_df, axis=1)
    unbalanced_mean = np.mean(unbalanced_df, axis=1)
    
    column_name = "class_bal_" + str(num_subintervals)
    balanced_dim_df[column_name] = balanced_mean
    unbalanced_dim_df[column_name] = unbalanced_mean
    

print(balanced_dim_df)
print(unbalanced_dim_df)
    
    


