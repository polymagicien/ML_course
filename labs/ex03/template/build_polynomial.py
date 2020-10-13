# -*- coding: utf-8 -*-
"""implement a polynomial basis function."""

import numpy as np


def build_poly(x, degree):
    """polynomial basis functions for input data x, for j=0 up to j=degree."""
    N = x.shape[0]
    base = np.ones(N)
    powered = np.ones(N)
    for _ in range(degree) :
        powered *= x
        base = np.concatenate((base, powered))
    return base.reshape(-1, N).transpose()
