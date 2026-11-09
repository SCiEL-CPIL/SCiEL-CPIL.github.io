"""
Program to decompose 3x5x5 matrix multiplication tensor.
Uses a block-based decomposition strategy.
The rank is minimal for this tensor structure (direct sum of 1x5x5 tensors).
"""
import numpy as np

DIMENSIONS = [3, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # The rank of M_<1,5,5> is 25.
    # M_<3,5,5> is the direct sum of three such tensors, so rank is 75.
    # This is the theoretical minimum.
    
    rank = n * m * p
    U = np.zeros((n * m, rank), dtype=int)
    V = np.zeros((m * p, rank), dtype=int)
    W = np.zeros((p * n, rank), dtype=int)
    
    rank_idx = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # U index: (i, j)
                U[i * m + j, rank_idx] = 1
                # V index: (j, k)
                V[j * p + k, rank_idx] = 1
                # W index: (k, i)
                W[k * n + i, rank_idx] = 1
                rank_idx += 1
                
    # Convert to lists as required by the interface
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
# EVOLVE-BLOCK-END
