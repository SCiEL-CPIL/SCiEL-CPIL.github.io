"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
import itertools
from fractions import Fraction

DIMENSIONS = [2, 4, 7]

# EVOLVE-BLOCK-START
import time

def construct():
    """Construct a valid tensor decomposition for 2x4x7 matrix multiplication.
    
    Returns the standard schoolbook decomposition with rank = n*m*p = 56.
    This is guaranteed to be correct and valid.
    """
    n, m, p = DIMENSIONS  # 2, 4, 7
    
    # Schoolbook decomposition: one term for each (i, j, k) triple
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # U index: i*m + j
                # V index: j*p + k
                # W index: k*n + i
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
