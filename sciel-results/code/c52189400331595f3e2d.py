"""Matrix multiplication 3x4 by 4x7: rank-minimal exact tensor decomposition."""
DIMENSIONS = [3, 4, 7]

# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    """Construct a rank-84 exact decomposition for 3x4 * 4x7 matrix multiplication (schoolbook)."""
    n, m, p = DIMENSIONS
    rank = n * m * p  # 84
    U = [[0] * rank for _ in range(n * m)]  # 12 rows
    V = [[0] * rank for _ in range(m * p)]  # 28 rows
    W = [[0] * rank for _ in range(p * n)]  # 21 rows
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j][col] = 1
                V[j*p+k][col] = 1
                W[k*n+i][col] = 1
                col += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
