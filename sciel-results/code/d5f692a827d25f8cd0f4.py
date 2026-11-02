"""Matrix multiplication 2x2 by 2x16: rank-minimal exact tensor decomposition."""
import time
import numpy as np

DIMENSIONS = [2, 2, 16]

# EVOLVE-BLOCK-START
def construct():
    # Schoolbook decomposition of rank n*m*p = 2*2*16 = 64
    n, m, p = DIMENSIONS
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
