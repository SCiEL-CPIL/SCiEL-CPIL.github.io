"""
Matrix multiplication 4x4x9 rank-minimal exact tensor decomposition.
The optimal rank is 144, achieved by the schoolbook algorithm.
"""
import numpy as np
import time

DIMENSIONS = [4, 4, 9]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank = n * m * p
    
    # Initialize U, V, W with zeros
    U = np.zeros((n * m, rank), dtype=np.int64)
    V = np.zeros((m * p, rank), dtype=np.int64)
    W = np.zeros((p * n, rank), dtype=np.int64)
    
    col = 0
    for i in range(n):
        for k in range(m):
            for j in range(p):
                u_idx = i * m + k
                v_idx = k * p + j
                w_idx = j * n + i
                
                U[u_idx, col] = 1
                V[v_idx, col] = 1
                W[w_idx, col] = 1
                col += 1
                
    return {
        'U': U,
        'V': V,
        'W': W
    }
# EVOLVE-BLOCK-END
