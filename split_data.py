# -*- coding: utf-8 -*-
"""Exercise 3.

Split the dataset based on the given ratio.
"""
import numpy as np

def split_data(x, y, ratio, seed=1):
    """
    split the dataset based on the split ratio. If ratio is 0.8 
    you will have 80% of your data set dedicated to training 
    and the rest dedicated to testing
    """
    print("x shape = ",x.shape)
    print("y shape = ",y.shape)
    # set seed
    np.random.seed(seed)
    shuffle = np.random.permutation(x.shape[0])
    # ***************************************************
    # INSERT YOUR CODE HERE
    # split the data based on the given ratio: TODO
    # ***************************************************
    stop = int(x.shape[0] * ratio)
    xT = x[shuffle][:stop]
    yT = y[shuffle][:stop]
    xV = x[shuffle][stop:]
    yV = y[shuffle][stop:]
    return xT, yT, xV, yV