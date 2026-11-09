# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    """
    Rank-minimal exact tensor decomposition for 4x4 * 4x7 matrix multiplication.
    
    Strategy: Schoolbook algorithm.
    Rank = 4 * 4 * 7 = 112.
    """
    n, m, p = DIMENSIONS  # 4, 4, 7
    
    rank = n * m * p
    
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, column] = 1
                V[j * p + k, column] = 1
                W[k * n + i, column] = 1
                column += 1
                
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
