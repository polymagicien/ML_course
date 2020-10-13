# -*- coding: utf-8 -*-
"""Exercise 3.

Ridge Regression
"""

import numpy as np

def compute_mse(y, tx, w):
    """Calculate the loss.

    You can calculate the loss using mse or mae.
    """
    N = y.shape[0]
    e = y - tx@w
    return 1/N * 0.5 * e.T@e 

def compute_rmse(y, tx, w):
    return np.sqrt(2 * compute_mse(y, tx, w))

def ridge_regression(y, tx, lambda_):
    """implement ridge regression."""
    N = y.shape[0]
    D = tx.shape[1]
    product = tx.T@tx
    w_optimal = np.linalg.solve(product + 2*N*lambda_*np.identity(D), tx.T@y)
    return w_optimal, compute_mse(y, tx, w_optimal)
