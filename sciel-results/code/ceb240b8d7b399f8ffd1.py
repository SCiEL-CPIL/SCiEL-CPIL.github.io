"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
from itertools import product

DIMENSIONS = [4, 5, 5]


# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    
    # Use the standard Kronecker product decomposition.
    # This is always valid and has rank n*m*p.
    # It serves as a baseline valid solution.
    R = n * m * p
    
    U = np.zeros((n * m, R))
    V = np.zeros((m * p, R))
    W = np.zeros((p * n, R))
    
    # The tensor T has non-zero entries at (i, j, k) for i in [0,n), j in [0,m), k in [0,p)
    # T[i, j, k] = 1
    # We decompose T = sum_r u_r v_r w_r^T (conceptually)
    # In the format requested:
    # U has shape (n*m, R)
    # V has shape (m*p, R)
    # W has shape (p*n, R)
    # The contraction is sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # This should equal T[i, j, k] = 1 if valid.
    
    idx = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # U index: i*m + j
                # V index: j*p + k
                # W index: k*n + i (following W[k*n+i, r] convention)
                
                U[i * m + j, idx] = 1.0
                V[j * p + k, idx] = 1.0
                W[k * n + i, idx] = 1.0
                
                idx += 1
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
