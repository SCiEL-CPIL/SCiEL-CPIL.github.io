"""Combined: schoolbook decomposition for 3x4x8 matrix multiplication tensor."""
import time
import numpy as np

DIMENSIONS = [3, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    """
    Construct an exact tensor decomposition for multiplying a 3x4 matrix
    by a 4x8 matrix using the schoolbook algorithm (rank 96).
    """
    deadline = time.time() + 110
    
    n, m, p = DIMENSIONS  # 3, 4, 8
    rank = n * m * p  # 96
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
