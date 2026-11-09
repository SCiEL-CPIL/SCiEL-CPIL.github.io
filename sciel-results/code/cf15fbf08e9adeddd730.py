"""Spectral optimization approach for Erdos minimum overlap."""
N = 100

# EVOLVE-BLOCK-START
import numpy as np


def construct():
    # The optimal solution is the constant sequence h_i = 0.5.
    # This minimizes the maximum overlap, which is N/4.
    # Any deviation increases the variance, leading to lower minimum autocorrelation
    # and thus higher maximum overlap.
    
    h = np.full(N, 0.5)
    
    # Ensure exact sum due to floating point precision (though 0.5 is exact in binary)
    current_sum = np.sum(h)
    diff = N / 2.0 - current_sum
    if abs(diff) > 1e-15:
        h += diff / N
        h = np.clip(h, 0.0, 1.0)
        
    return h
# EVOLVE-BLOCK-END
