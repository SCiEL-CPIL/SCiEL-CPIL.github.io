import time
import numpy as np
import sys

DIMENSIONS = [3, 3, 5]

# EVOLVE-BLOCK-START
def construct():
    """
    Returns a rank-45 exact tensor decomposition for 3x3 * 3x5 matrix multiplication.
    Uses schoolbook multiplication.
    """
    n, m, p = DIMENSIONS  # 3, 3, 5
    rank = n * m * p  # 45
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_row = i * m + j
                v_row = j * p + k
                w_row = k * n + i
                U[u_row][col] = 1
                V[v_row][col] = 1
                W[w_row][col] = 1
                col += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
