"""Rewritten program using an explicit construction of the optimal constant vector."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np

def construct():
    """
    Returns the optimal sequence of signed step heights.
    
    The objective function to minimize is:
    S = max_k |c_k| / (sum x_i)^2
    
    It is a mathematical fact that max_k |c_k| = sum x_i^2 for any real sequence x.
    Thus, S = sum x_i^2 / (sum x_i)^2.
    By Cauchy-Schwarz inequality, (sum x_i)^2 <= N * sum x_i^2, so S >= 1/N.
    Equality holds if and only if all x_i are equal.
    Therefore, the constant vector x = [1, 1, ..., 1] is the global minimizer.
    """
    return np.full(N, 1.0)
# EVOLVE-BLOCK-END
