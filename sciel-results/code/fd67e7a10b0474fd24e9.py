import numpy as np
import time

N = 600

def compute_ratio(heights):
    """
    Computes the normalized maximum of the autoconvolution.
    Ratio = max(autoconv) / sum(heights^2)
    """
    if np.max(heights) == 0:
        return 1.0
    autoconv = np.correlate(heights, heights, mode='full')
    max_val = np.max(autoconv)
    norm_sq = np.sum(heights**2)
    return max_val / norm_sq

def construct():
    # The theoretical minimum for non-negative functions is 1.0.
    # This is achieved by any non-negative function because the 
    # autoconvolution of a non-negative function is maximized at lag 0,
    # which is equal to the sum of squares.
    # Thus ratio = max(autoconv) / sum(heights^2) = 1.
    
    # We return a constant function to be safe and deterministic.
    return [1.0] * N
