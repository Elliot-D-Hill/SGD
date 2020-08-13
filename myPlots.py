#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May  7 21:19:04 2020

@author: Elliot
"""
import numpy as np 
import seaborn as sns
import matplotlib.pyplot as plt


def plot_data(cost_function, num_points, mini_batch_size, data):
    
    if cost_function.method == 'LS':
        
        x1 = data[:num_points, 1]
        y1 = data[:num_points, 2]
        
        plt.scatter(x1, y1, marker = '.')
        idx = np.random.choice(data.shape[0], mini_batch_size, replace=False)

        mini_batch = data[idx,:]
        
        plt.scatter(mini_batch[:mini_batch_size, 1], 
                    mini_batch[:mini_batch_size, 2], 
                    marker = '.',
                    color = 'red')
                
        plt.ylabel("y")
        plt.xlabel("x") 

        # plt.show() 
        
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
        plt.plot(error_list) 
        plt.xlabel("Number of iterations") 
        plt.ylabel("Cost") 
        plt.show() 
    
    elif cost_function.method == 'LR':
        sns.set_style('white')
        plt.plot(range(len(error_list)), error_list, 'r')
        plt.xlabel("Number of Iterations")
        plt.ylabel("Cost")
        plt.show()


def ecdf(data):
        x = np.sort(data)
        x = np.log(x)
        n = x.shape[0]
        y = np.arange(1, n+1) / n
        return(x, y)

# myPlots.plot_ecdf(balanced_dim_df, unbalanced_dim_df, 2, 3, dimensions)

def plot_ecdf(data1, data2, num_row_plots, num_col_plots, exp_var):

    fig, axs = plt.subplots(num_row_plots, num_col_plots)
    
    count = 0
    for i in range(num_row_plots):
        for j in range(num_col_plots):
            
            d1 = data1.iloc[:, count]
            d2 = data2.iloc[:, count]
            
            x1, y1 = ecdf(d1)
            x2, y2 = ecdf(d2)
            axs[i, j].plot(x1, y1, 'b', label='Balanced')
            axs[i, j].plot(x2, y2, 'orange', label='Unbalanced')
            axs[i, j].set_title('Class imbalance: ' + str(exp_var[count]), fontsize = 12)

            count += 1
    
    for i, ax in enumerate(axs.flat):
        # ax.set(xlabel='Sample evaluations', ylabel='Fraction of target values reached')
        if i >= 3:
            ax.set_xlabel('log(sample evaluations)', fontsize = 12)
        if i == 0 or i == 3:
            ax.set_ylabel('Fraction of targets reached', fontsize = 12)
        
    plt.tight_layout()
    plt.show()
    
def plot_single_ecdf(data):
    
    color_palette = sns.color_palette("OrRd", 12)
    
    for i, col in enumerate(data.columns):
        d = data[col]            
        x, y = ecdf(d)
        plt.plot(x, y, label=col, color=color_palette[11-i*2])
    
    plt.legend(title="Class imbalance")
    plt.xlabel('log(sample evaluations)', fontsize = 16)
    plt.ylabel('Fraction of targets reached', fontsize = 16)
    
    
    