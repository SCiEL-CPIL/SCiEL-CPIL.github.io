"""Tensor decomposition for 3x3x7 matrix multiplication."""
import numpy as np
from itertools import product

DIMENSIONS = [3, 3, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = 3, 3, 7
    # The tensor T[i,j,k] has a nonzero entry at (i*m+j, j*p+k, k*n+i)
    # Trivial decomposition: rank = n*m*p = 63
    # Each term is a rank-1 tensor: U_col * V_col * W_col
    # where U has 1 at (i*m+j), V has 1 at (j*p+k), W has 1 at (k*n+i)
    
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1.0
                V[j * p + k, col] = 1.0
                W[k * n + i, col] = 1.0
                col += 1
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
