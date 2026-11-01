"""Matrix multiplication 2x3x6 rank-minimal decomposition."""
DIMENSIONS = [2, 3, 6]

# EVOLVE-BLOCK-START
import numpy as np
import time

def construct():
    """
    Construct rank-minimal decomposition for 2x3x6 matrix multiplication.
    Uses schoolbook algorithm with rank 36.
    """
    n, m, p = 2, 3, 6
    rank = n * m * p  # 36
    
    U = np.zeros((n * m, rank))  # (6, 36)
    V = np.zeros((m * p, rank))  # (18, 36)
    W = np.zeros((p * n, rank))  # (12, 36)
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
