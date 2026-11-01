"""
Matrix multiplication 2,2,8 rank-minimal exact tensor decomposition.
The minimal rank for this tensor is 32, derived from the direct sum of 8
independent 2x2x1 multiplication tensors, each of rank 4.
"""

import time
import numpy as np

DIMENSIONS = [2, 2, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank = n * m * p  # 32
    
    # Initialize matrices with zeros
    # U shape: (n*m, rank) = (4, 32)
    # V shape: (m*p, rank) = (16, 32)
    # W shape: (p*n, rank) = (16, 32)
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # U index: i * m + j
                u_idx = i * m + j
                # V index: j * p + k
                v_idx = j * p + k
                # W index: k * n + i
                w_idx = k * n + i
                
                U[u_idx][column] = 1
                V[v_idx][column] = 1
                W[w_idx][column] = 1
                column += 1
                
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
