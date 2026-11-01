"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import itertools
from collections import defaultdict

DIMENSIONS = [2, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Rank 5 decomposition
    rank = m  # 5
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # U[i*m + j, c] = 1 if j == c, else 0
    for i in range(n):
        for j in range(m):
            if j < rank:
                U[i * m + j][j] = 1
    
    # V[j*p + k, c] = 1 if j == c, else 0
    for j in range(m):
        for k in range(p):
            if j < rank:
                V[j * p + k][j] = 1
    
    # W[k*n + i, c] = 1 for all c
    for k in range(p):
        for i in range(n):
            for c in range(rank):
                W[k * n + i][c] = 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
