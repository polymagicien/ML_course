# -*- coding: utf-8 -*-
"""Exercise 3.

Split the dataset based on the given ratio.
"""


import numpy as np


def split_data(x, y, ratio, seed=1):
    """split the dataset based on the split ratio."""
    # set seed
    np.random.seed(seed)

    N = x.shape[0]
    index_max = int(ratio * N)
    indexes = np.random.permutation(np.arange(N))
    x_shuffled = x[indexes]
    y_shuffled = y[indexes]

    x_train = x_shuffled[:index_max]
    y_train = y_shuffled[:index_max]
    x_test = x_shuffled[index_max:]
    y_test = y_shuffled[index_max:]

    return x_train, y_train, x_test, y_test