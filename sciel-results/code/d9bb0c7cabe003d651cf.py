import time
import sys
import numpy as np

DIMENSIONS = [2, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 115
    
    n, m, p = DIMENSIONS
    
    # Rank is m
    total_rank = m
    
    U = np.zeros((n * m, total_rank), dtype=float)
    V = np.zeros((m * p, total_rank), dtype=float)
    W = np.zeros((p * n, total_rank), dtype=float)
    
    # U[i*m + j, r] = 1 if j == r else 0
    for i in range(n):
        for j in range(m):
            U[i * m + j, j] = 1.0
            
    # V[j*p + k, r] = 1 if j == r else 0
    for j in range(m):
        for k in range(p):
            V[j * p + k, j] = 1.0
            
    # W[k*n + i, r] = 1 for all k, i, r
    W.fill(1.0)
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
