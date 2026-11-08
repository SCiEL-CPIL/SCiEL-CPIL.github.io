"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from scipy.optimize import minimize

DIMENSIONS = [2, 3, 6]

# EVOLVE-BLOCK-START
def construct():
    """
    Returns a valid rank-36 decomposition for 2x3x6 matrix multiplication.
    This uses the trivial decomposition where each product term A_ij B_jk is computed separately.
    """
    n, m, p = DIMENSIONS
    R = n * m * p
    
    U = np.zeros((n * m, R))
    V = np.zeros((m * p, R))
    W = np.zeros((p * n, R))
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                r = i * (m * p) + j * p + k
                # U index: (i, j) -> i*m + j
                U[i * m + j, r] = 1.0
                # V index: (j, k) -> j*p + k
                V[j * p + k, r] = 1.0
                # W index: (k, i) -> k*n + i
                W[k * n + i, r] = 1.0
                
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
