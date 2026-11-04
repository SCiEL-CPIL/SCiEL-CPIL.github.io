"""Tensor decomposition for 4x4 x 4x6 matrix multiplication.

Uses schoolbook decomposition with rank 96 (4*4*6).
This is correct but not optimal. A better decomposition would use lower rank.
"""

import time


def construct():
    """
    Returns a dictionary with 'U', 'V', 'W' factor matrices.
    
    U: shape (16, rank)
    V: shape (24, rank)
    W: shape (24, rank)
    """
    start_time = time.time()
    
    n, m, p = 4, 4, 6
    rank = n * m * p  # Schoolbook rank = 96
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    r_idx = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][r_idx] = 1
                V[j * p + k][r_idx] = 1
                W[k * n + i][r_idx] = 1
                r_idx += 1
    
    elapsed = time.time() - start_time
    if elapsed > 140:
        pass
    
    return {"U": U, "V": V, "W": W}
