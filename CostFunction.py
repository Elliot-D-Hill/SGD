#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 16 10:31:52 2020

@author: Elliot
"""
import numpy as np
from numpy import linalg as LA

class CostFunction:
    
    def __init__(self, method, grad_method):
        self.method = method
        self.grad_method = grad_method

    def sigmoid(self, y_hat): # for logistic regression
        return 1 / (1 + np.exp(-y_hat))
    
    # define model
    def model(self, X, theta):
        y_hat = np.dot(X, theta)
        if self.method == 'LS': # least squares
             None # do nothing       
        if self.method == 'LR': # logistic regression
            y_hat = self.sigmoid(y_hat)
        return y_hat
    
    # calculate the residual
    def residual(self, X, theta, y):
        y_hat = self.model(X, theta)
        R =  y_hat - y
        return R
    
    # calculate the Jacobian
    def jacobian(self, X, y, theta):
        n = len(y)
        if self.method == 'LS': # least squares
            R = self.residual(X, theta, y)
            J = (1 / n) * LA.norm(R) # R^T * R
            
        if self.method == 'LR': # logistic regression
            y_hat = self.model(X, theta)
            epsilon = 1e-5
            J = (1 / n) * (((-y).T @ np.log(y_hat + epsilon))
                       -((1 - y).T @ np.log(1 - y_hat + epsilon)))
            J = J[0]
        return J
    
    
    def gradient(self, X, y, theta):
        
        # analytcal gradient of normal equations w.r.t. theta 
        if self.grad_method == 'exact': 
            R = self.residual(X, theta, y)
            grad = np.dot(X.transpose(), R) 
            
        # finite difference approximation of gradient
        elif self.grad_method == 'fdm': 
            n = theta.size
            grad = np.empty(theta.shape, dtype=float)
            d_theta = 10**-4
            theta_diff = theta
            
            for i in range(n):
                theta_diff[i] = theta[i] + d_theta
                fp = self.jacobian(X, y, theta_diff)
                theta_diff[i] = theta[i] - d_theta
                fm = self.jacobian(X, y, theta_diff)
                grad[i] = (fp - fm) / (2 * d_theta)
                theta_diff[i] = theta[i]
        return np.array(grad) 