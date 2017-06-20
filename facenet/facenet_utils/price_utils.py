# -*- coding: utf-8 -*-
"""
Created on Mon May 22 20:07:49 2017

@author: alexandre
"""


import numpy as np
import math
import time
import cv2
import pandas as pd


def set_commat_to_price(vector_price):
        
        def apply_min_diff_ref_mean(x):
            if len(x) >0: 
                if x[-2:] == '99' and len(x)>2:
                    x = float(x[:-2] +'.'+x[-2:])
                else:
                    index = len(x)
                    mini = abs(float('0.' + x) - reference_mean)
                    for i in range(1, len(x)+1):
                        y = float(x[:i] +'.'+x[i:])
                        
                        if abs(y-reference_mean) < mini:
                            index = i
                           
                    x = float(x[:index] +'.'+x[index:])
            return x
        
        prices_99 = vector_price[vector_price.apply(lambda x : x[-2:] == '99' or x[-2:] == '95' or x[-2:] == '90')]
        
        if len(prices_99) >0:
            prices_99 = prices_99.apply(lambda x : x[:-2] +'.'+x[-2:]).astype(float)
            reference_mean= np.median(prices_99)

        vector_price = vector_price.apply(lambda x : apply_min_diff_ref_mean(x))
        return vector_price
        
        