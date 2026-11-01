"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import itertools
from collections import defaultdict

DIMENSIONS = [2, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank = n * m
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for i in range(n):
        for r in range(m):
            r_idx = i * m + r
            # U[i, r] = 1
            U[i * m + r][r_idx] = 1
            # V[r, k] = 1 for all k
            for k in range(p):
                V[r * p + k][r_idx] = 1
            # W[k, i] = 1 for all k
            for k in range(p):
                W[k * n + i][r_idx] = 1
                
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
