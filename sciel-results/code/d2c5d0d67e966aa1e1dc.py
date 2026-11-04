"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import sys
import numpy as np

DIMENSIONS = [3, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    """
    Construct a rank-minimal exact tensor decomposition for 3x5 @ 5x5 matrix multiplication.
    
    Uses the schoolbook construction with rank = n*m*p = 75.
    Each rank-1 term corresponds to a unique (i, j, k) triple where:
    - i in [0, n): row index of A / C
    - j in [0, m): column index of A / row index of B  
    - k in [0, p): column index of B / C
    
    The trace convention for W is W[k*n + i, r].
    """
    n, m, p = DIMENSIONS
    rank = n * m * p  # 75
    
    U = np.zeros((n * m, rank), dtype=np.int64)
    V = np.zeros((m * p, rank), dtype=np.int64)
    W = np.zeros((p * n, rank), dtype=np.int64)
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
# EVOLVE-BLOCK-END
