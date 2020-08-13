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
import myPlots


############# experimental parameters that vary

dimensions = [2, 4, 8, 16, 32, 64]
class_imbalance = np.around(np.arange(start=0.1, stop=0.6, step=0.1), decimals=1)
class_imbalance = np.insert(class_imbalance, 0, 0.01)
mini_balance = class_imbalance
subintervals = [4, 8, 16, 32, 64, 128]   

############# experimental parameters that are fixed

num_problems = 30
sample_size = 4000
num_targets = 20
num_subintervals = 64
dim = 32
balance = 0.5
noise = 25

############# gradient descent parameters 

parms = pm.Parameters(learning_rate = 0.001,
                      batch_size = 64, 
                      tol = 0.01,
                      max_epoch = 10,
                      max_iterations = 200,
                      decay = 1, 
                      decay_schedule = 1024,
                      balanced = True)

############# choose cost function 

cost_function = cf.CostFunction(method = 'LS', # choose from: 'LS' or 'LR'
                                grad_method = 'exact') # choose: 'exact' or 'fdm'

experiment_var = dimensions

############# Create cost function targets values

random_seed = 2

def run_experiment(experiment_var, random_seed):
    
    evals_at_targets_df = pd.DataFrame()
    
    for i, dim in enumerate(experiment_var):
        
        a = 1
        b = -1
        # random initial solution with elements between -1 and 1
        theta0 = (b - a) * np.random.rand(dim+1, 1) + a
        
        # allow more iterations in higher dimensions
        parms.max_iterations = parms.max_iterations * dim
    
        error_list, sample_evals = ex.run_problem(dim, 
                                                  sample_size, 
                                                  num_targets,
                                                  num_subintervals,
                                                  cost_function,
                                                  theta0,
                                                  balance,
                                                  noise,
                                                  parms,
                                                  random_seed)
               
        ############# benchmark optimization run
        
        target_values = ex.create_targets(error_list, num_targets)
        
        benchmarker = bm.Benchmark(sample_evals, target_values, error_list)
            
        evals_at_targets = benchmarker.benchmark()
        evals_at_targets_df[i] = evals_at_targets
          
    return evals_at_targets_df
    
    # test error
    # y_pred, err, majority_err, minority_err = bm.test_error(cost_function, X_test, y_test, theta, verbose = True)
    

evals_at_targets_df = run_experiment(experiment_var, random_seed)
print(evals_at_targets_df)
    
# myPlots.plot_ecdf(mean_df, 2, 3, exp_var)


