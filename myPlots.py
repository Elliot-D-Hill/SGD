#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May  7 21:19:04 2020

@author: Elliot
"""
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np 
from statsmodels.distributions.empirical_distribution import ECDF

font = {'size' : 14}
plt.rc('font', **font)

def plot_data(cost_function, numPoints, data):
    
    if cost_function.method == 'LS':
        plt.scatter(data[:numPoints, 1], data[:numPoints, 2], marker = '.') 
        plt.show() 
    elif cost_function.method == 'LR':
        sns.set_style('white')
        sns.scatterplot(data[:,1], data[:,2], hue=data[:,3]);

def plot_model_fit(cost_function, X_test, y_test, y_pred,  theta):
    
    if cost_function.method == 'LS':
        plt.scatter(X_test[:, 1], y_test[:, ], marker = '.') 
        plt.plot(X_test[:, 1], y_pred, color = 'orange')
        plt.xlabel("x") 
        plt.ylabel("y")
        plt.show() 
        
    elif cost_function.method == 'LR':
        slope = -(theta[1] / theta[2])
        intercept = -(theta[0] / theta[2])
        sns.set_style('white')
        sns.scatterplot(X_test[:,1], X_test[:,2], hue=y_test.reshape(-1));
        ax = plt.gca()
        ax.autoscale(False)
        x_vals = np.array(ax.get_xlim())
        y_vals = intercept + (slope * x_vals)
        plt.plot(x_vals, y_vals, c="k");

def plot_cost(cost_function, error_list):
    if cost_function.method == 'LS':
        plt.figure()
        plt.plot(error_list) 
        plt.xlabel("Number of iterations") 
        plt.ylabel("Cost") 
        plt.show() 
    
    elif cost_function.method == 'LR':
        plt.figure()
        sns.set_style('white')
        plt.plot(range(len(error_list)), error_list, 'r')
        plt.xlabel("Number of Iterations")
        plt.ylabel("Cost")
        plt.show()

def plot_ecdf(data, p_features):
    
    def ecdf(data):
        x = np.sort(data)
        x = np.log(x/p_features)
        x = x/p_features
        n = x.size
        y = np.arange(1, n+1) / n
        return(x,y)
    
    x, y = ecdf(data)
    plt.figure()
    plt.plot(x, y)
    plt.xlabel("log(sample evaluations / dimension)")
    plt.ylabel("Fraction of target values reached")