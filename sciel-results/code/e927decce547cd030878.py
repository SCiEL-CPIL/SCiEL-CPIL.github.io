"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [2, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    _start = time.time()
    _deadline = _start + 140.0
    
    n, m, p = DIMENSIONS
    
    # The rank of M_{n,m,p} is at most m.
    # Decomposition:
    # U[i*m + j, r] = 1 if j == r else 0
    # V[j*p + k, r] = 1 if j == r else 0
    # W[k*n + i, r] = 1 for all k, i, r
    
    rank = m
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[1] * rank for _ in range(p * n)]
    
    for i in range(n):
        for j in range(m):
            r = j
            U[i * m + j][r] = 1
            V[j * p][r] = 1 # Wait, V index is j*p + k. We need to set for all k.
            
    # Correcting V construction
    for j in range(m):
        r = j
        for k in range(p):
            V[j * p + k][r] = 1
            
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
