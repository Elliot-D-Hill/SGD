#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 22 19:23:22 2020

@author: Elliot
"""

import numpy as np
from numpy import linalg as LA

class update_step:
    
    def __init__(self, theta, learning_rate):
        
        self.theta = theta
        self.learning_rate = learning_rate

class gradient_descent(update_step):
    
    def __init__(self, theta, learning_rate):
        super().__init__(theta, learning_rate)
        
    def update(self, grad):
        
        self.theta = self.theta - self.learning_rate * grad
        
        return self.theta

class momentum(update_step):
    
    def __init__(self, theta, learning_rate, velocity, gamma):
        
        super().__init__(theta, learning_rate)
        self.velocity = velocity
        self.gamma = gamma
    
    def update(self, grad):
        
        self.velocity = (self.gamma * self.velocity) + (self.learning_rate * grad)
        self.theta = self.theta - self.velocity
        
        return self.theta
    
class scaled_momentum(momentum):
    
    prev_grad = None
    
    def __init__(self, theta, learning_rate, velocity, gamma):
        super().__init__(theta, learning_rate, velocity, gamma)
        self.iteration = 1
        self.prev_grad
        self.gamma0 = gamma
    
    def update(self, grad):
        
        if self.iteration == 1:
            self.prev_grad = grad

        # FIXME don't really know what this stuff does
        grad_factor = LA.norm(grad) / LA.norm(self.prev_grad)
        iteration_factor = (self.iteration - 1) / self.iteration
        # iteration_factor = 1/self.iteration        

        # FIXME don't really know what this stuff does
        self.gamma = np.amin([iteration_factor, grad_factor]) * self.gamma0
        
        # momentum update step
        self.velocity = (self.gamma * self.velocity) + (self.learning_rate * grad)
        self.theta = self.theta - self.velocity
        
        self.prev_grad = grad
        self.iteration += 1

        
        return self.theta
    
class nesterov(momentum):
    
    def __init__(self, theta, learning_rate, velocity, gamma):
        super().__init__(theta, learning_rate, velocity, gamma)
    
    def update(self, grad):
        # FIXME this is just vanilla momentum right not
        self.velocity = (self.gamma * self.velocity) + (self.learning_rate * grad)
        self.theta = self.theta - self.velocity

class adam(nesterov):
    
    def __init__(self, theta, learning_rate, velocity, gamma):
        super().__init__(theta, learning_rate, velocity, gamma)
    
    def update(self):
        pass
    
class predict_correct(momentum):
    
    def __init__(self, theta, learning_rate, velocity, gamma):
        super().__init__(theta, learning_rate, velocity, gamma)
    
    def update(self, grad):
        # FIXME no idea if this works
        self.theta = self.theta - self.learning_rate * grad
        self.velocity = (self.gamma * self.velocity) + (self.learning_rate * grad)
        self.theta = self.theta - self.velocity
        
        return self.theta