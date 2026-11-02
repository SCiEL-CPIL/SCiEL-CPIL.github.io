import time
import numpy as np

DIMENSIONS = [2, 3, 7]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110
    
    # Target tensor T[i, j, k] = delta_{j, k}
    # We propose rank R = m * n = 3 * 2 = 6
    R = m * n
    
    U = [[0] * R for _ in range(n * m)]
    V = [[0] * R for _ in range(m * p)]
    W = [[0] * R for _ in range(p * n)]
    
    # For each i in [0, n-1], we use m ranks to form the identity matrix I_{m x p}
    # U[i, j, r] and V[j, k, r] such that sum_r U[i, j, r] V[j, k, r] = delta_{j, k}
    # We can set U[i, j, r] = 1 if r corresponds to (i, j)
    # And V[j, k, r] = 1 if k == j and r corresponds to (i, j)
    # W[k, i, r] = 1 for all k, i, r in the block for i
    
    col = 0
    for i in range(n):
        for j in range(m):
            r = col
            # U[i*m + j, r] = 1
            U[i * m + j][r] = 1
            # V[j*p + k, r] = 1 if k == j
            for k in range(p):
                if k == j:
                    V[j * p + k][r] = 1
            # W[k*n + i, r] = 1 for all k
            for k in range(p):
                W[k * n + i][r] = 1
            col += 1
            
    return {"U": U, "V": V, "W": W}
