import numpy as np
import random

DIMENSIONS = [3, 3, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Rank minimal decomposition for M_{n x m} * M_{m x p}
    # We can achieve rank n*m by grouping terms by (i, j)
    rank = n * m
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    r = 0
    for i in range(n):
        for j in range(m):
            # U[i*m + j, r] = 1
            U[i * m + j][r] = 1
            # V[j*p + k, r] = 1 for all k
            for k in range(p):
                V[j * p + k][r] = 1
                # W[k*n + i, r] = 1 for all k
                W[k * n + i][r] = 1
            r += 1
            
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
