"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np

def construct():
    N = 400
    
    # The objective is to minimize V = max_k |c_k| / (sum_i x_i)^2.
    # The theoretical lower bound is 1/N, achieved if and only if x_i are constant integers.
    # We directly construct the optimal constant signal.
    
    return np.ones(N, dtype=int)
# EVOLVE-BLOCK-END
