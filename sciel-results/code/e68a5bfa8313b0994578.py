"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

DIMENSIONS = [3, 4, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Schoolbook rank
    rank_schoolbook = n * m * p
    
    # Try to find a lower rank decomposition
    # The tensor T[i*m+j, j*p+k, k*n+i] = 1 for valid (i,j,k)
    # We want U, V, W such that sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1
    
    # Start with schoolbook as fallback
    U = np.zeros((n*m, rank_schoolbook))
    V = np.zeros((m*p, rank_schoolbook))
    W = np.zeros((p*n, rank_schoolbook))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                c = i*m*p + j*p + k
                U[i*m+j, c] = 1
                V[j*p+k, c] = 1
                W[k*n+i, c] = 1
    
    # Try to reduce rank using numerical optimization
    # This is a hard NP-hard problem in general, but for small cases we can try
    
    # For now, return schoolbook. Future work: try rank reduction.
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
