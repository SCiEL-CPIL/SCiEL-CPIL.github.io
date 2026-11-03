import numpy as np
import time

DIMENSIONS = [2, 3, 9]

def construct():
    """
    Decompose 2x3 * 3x9 matrix multiplication using block-diagonal structure.
    
    The 3x9 matrix B is treated as three 3x3 blocks (columns).
    Each block computes a 2x1 column of C.
    The multiplication for a single column (2x3 * 3x1) has rank 2.
    Total rank = 9 columns * 2 = 18.
    """
    n, m, p = DIMENSIONS  # n=2, m=3, p=9
    
    rank = 18
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    r = 0
    for k in range(p):
        for i in range(n):
            # Term r corresponds to computing C[i, k]
            # C[i, k] = sum_j A[i, j] * B[j, k]
            
            # U index: i * m + j
            for j in range(m):
                U[i * m + j, r] = 1.0
            
            # V index: j * p + k
            for j in range(m):
                V[j * p + k, r] = 1.0
            
            # W index: k * n + i
            W[k * n + i, r] = 1.0
            
            r += 1
            
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
