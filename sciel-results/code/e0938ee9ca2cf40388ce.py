"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 4, 7]

# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    n, m, p = DIMENSIONS
    # Rank m * n using the independent row decomposition
    # For each i in 0..n-1, we use m terms.
    # Total rank = n * m
    rank = n * m
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    for i in range(n):
        for j in range(m):
            # U[i*m + j, col] = 1
            U[i * m + j][col] = 1
            # V[j*p + k, col] = 1 for all k
            for k in range(p):
                V[j * p + k][col] = 1
            # W[k*n + i, col] = 1 for all k
            for k in range(p):
                W[k * n + i][col] = 1
            col += 1
            
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
