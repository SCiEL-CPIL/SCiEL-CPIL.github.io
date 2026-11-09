import time
import numpy as np

DIMENSIONS = [2, 4, 4]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140.0
    
    rank = 8
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Rank 8 decomposition
    # For each row i in {0, 1} and each column j in {0, 1, 2, 3}
    # We create a term r = i * 4 + j
    # u_r has 1 at (i, j)
    # v_r has 1 at (j, k) for all k in {0, 1, 2, 3}
    # w_r has 1 at (k, i) for all k in {0, 1, 2, 3}
    
    for i in range(n):
        for j in range(m):
            r = i * m + j
            
            # U: index i*m + j
            U[i * m + j][r] = 1
            
            # V: indices j*p + k for k in 0..p-1
            for k in range(p):
                V[j * p + k][r] = 1
            
            # W: indices k*n + i for k in 0..p-1
            for k in range(p):
                W[k * n + i][r] = 1
                
    return {"U": U, "V": V, "W": W}
