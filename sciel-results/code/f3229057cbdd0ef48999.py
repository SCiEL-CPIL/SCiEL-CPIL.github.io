"""Rewritten with vectorized construction for matrix multiplication tensor."""
DIMENSIONS = [2, 3, 10]

import numpy as np

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank = n * m * p
    
    # Generate all combinations of (i, j, k)
    i_idx, j_idx, k_idx = np.meshgrid(np.arange(n), np.arange(m), np.arange(p), indexing='ij')
    
    # Calculate row indices for U, V, W
    # U index: i * m + j
    # V index: j * p + k
    # W index: k * n + i
    u_rows = (i_idx * m + j_idx).flatten()
    v_rows = (j_idx * p + k_idx).flatten()
    w_rows = (k_idx * n + i_idx).flatten()
    
    cols = np.arange(rank)
    
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    U[u_rows, cols] = 1
    V[v_rows, cols] = 1
    W[w_rows, cols] = 1
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
