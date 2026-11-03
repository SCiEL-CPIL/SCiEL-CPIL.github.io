"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 100

# EVOLVE-BLOCK-START
import numpy as np

def construct():
    """
    Returns the optimal sequence of step heights for the Erdos minimum overlap problem.
    
    The overlap for shift k is O_k = sum_i h[i] * (1 - h[(i+k)%N]).
    Summing over all k: sum_k O_k = sum_i h[i] * (N - sum_j h[j]) = (N/2) * (N/2) = N^2/4.
    Average overlap = N/4. Thus max overlap >= N/4.
    
    Uniform h[i] = 0.5 gives O_k = N * 0.25 = N/4 for all k, achieving the lower bound.
    Therefore uniform 0.5 is the global optimum.
    """
    return np.full(N, 0.5)
# EVOLVE-BLOCK-END
