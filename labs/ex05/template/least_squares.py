# -*- coding: utf-8 -*-
"""Exercise 3.

Least Square
"""

import numpy as np

def compute_mse(y, tx, w):
    """Calculate the loss.

    You can calculate the loss using mse or mae.
    """
    N = y.shape[0]
    e = y - tx@w
    return 1/N * 0.5 * e.T@e 

    
def least_squares(y, tx):
    """calculate the least squares."""
    w_optimal = np.linalg.solve(tx.T@tx, tx.T@y)
    return compute_mse(y, tx, w_optimal), w_optimal
