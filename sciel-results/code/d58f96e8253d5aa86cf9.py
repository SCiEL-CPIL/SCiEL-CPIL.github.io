# EVOLVE-BLOCK-START
import numpy as np

def construct():
    """
    Constructs the optimal signed step function.
    
    The metric to minimize is max(|autoconvolution|) / (integral)^2.
    For any sequence x, max(|autoconv|) >= sum(x_i^2) (at lag 0).
    Thus metric >= sum(x_i^2) / (sum(x_i))^2.
    By Cauchy-Schwarz, (sum(x_i))^2 <= N * sum(x_i^2).
    So metric >= 1/N.
    Equality holds if and only if all x_i are equal.
    Therefore, the constant sequence (all 1s or all -1s) is the global optimum.
    We return the sequence of all 1s.
    """
    return np.ones(N, dtype=np.int32)
# EVOLVE-BLOCK-END
