"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np

def construct():
    """
    Returns the optimal constant sequence.
    
    The objective is to minimize V = max_k |C[k]| / (sum_i f_i)^2.
    For any real sequence f, (sum f)^2 <= N * sum(f^2) = N * C[0] (Cauchy-Schwarz).
    Also max_k |C[k]| >= |C[0]| = sum(f^2).
    Therefore, V >= sum(f^2) / (sum f)^2 >= sum(f^2) / (N * sum(f^2)) = 1/N.
    The constant sequence f_i = c achieves equality in both inequalities,
    resulting in V = 1/N.
    """
    return np.full(N, 1.0)
# EVOLVE-BLOCK-END
