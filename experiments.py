#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May  8 15:26:38 2020

@author: Elliot
"""
import numpy as np


dimensions = [1, 2, 4, 8, 16, 32, 64]
class_imbalance = np.around(np.arange(start=0.1, stop=0.6, step=0.1), decimals=1)
mini_balance = class_imbalance
subintervals = [1, 2, 4, 8, 16, 32]

target_values = np.around(np.arange(start=0.1, stop=1, step=0.1), decimals=1)    


for dim in dimensions:
    print(dim)

for imbalance in class_imbalance:
    print(imbalance)
    
for balance in mini_balance:
    print(balance)
    
for subinterval in subintervals:
    print(subinterval)
    
    


