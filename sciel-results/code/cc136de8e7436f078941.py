import time
import sys
import numpy as np

DIMENSIONS = [2, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    n, m, p = DIMENSIONS
    
    # The tensor T[i,j,k] = 1 for all i,j,k.
    # We can decompose this into m rank-1 tensors, one for each j.
    # For each j, U[i,j,r_j] = 1, V[j,k,r_j] = 1, W[k,i,r_j] = 1.
    # This gives rank m = 4.
    
    rank = m
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for j in range(m):
        r = j
        # Set U[i, j, r] = 1 for all i
        for i in range(n):
            U[i * m + j][r] = 1
        # Set V[j, k, r] = 1 for all k
        for k in range(p):
            V[j * p + k][r] = 1
        # Set W[k, i, r] = 1 for all k, i
        for k in range(p):
            for i in range(n):
                W[k * n + i][r] = 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
