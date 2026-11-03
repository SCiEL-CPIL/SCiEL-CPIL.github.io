import numpy as np

def construct():
    """
    Returns step heights minimizing the maximum overlap of h with shifted (1-h).
    
    The problem is to minimize max_k C_k where C_k = sum_i h_i (1-h_{i+k}).
    Given sum h_i = N/2, the sum of all C_k for k=1..N-1 is 
    (N-2 - sum h_i) * sum h_i + sum h_i^2 = (N-2 - N/2)*(N/2) + sum h_i^2.
    For N=100, sum C_k = 49*50 + sum h_i^2 = 2450 + sum h_i^2.
    To minimize the maximum, we minimize the sum, which requires minimizing sum h_i^2.
    This is achieved by uniform h_i = 0.5.
    This yields C_k = 25 for all k, which is the theoretical minimum.
    """
    N = 100
    h = np.full(N, 0.5)
    return h
