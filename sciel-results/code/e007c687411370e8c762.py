"""Rewritten program for the suite task; returns the optimal constant vector."""
import numpy as np

N = 100

# EVOLVE-BLOCK-START
def construct():
    """
    The objective is to minimize the maximum overlap O(s) = sum(h_i * (1 - h_{i+s})).
    The average overlap over all circular shifts is exactly N/4 = 25 for N=100.
    The constant vector h_i = 0.5 achieves O(s) = 25 for all s.
    Since Max >= Average, 25 is the global minimum.
    We return this optimal solution directly.
    """
    return np.full(N, 0.5)
# EVOLVE-BLOCK-END
