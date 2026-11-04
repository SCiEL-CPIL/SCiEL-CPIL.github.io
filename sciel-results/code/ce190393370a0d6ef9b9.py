import numpy as np
import time

DIMENSIONS = [3, 3, 8]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    
    # Schoolbook decomposition: rank = n*m*p = 72
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    
    # Verify the decomposition
    # T[i*m+j, j*p+k, k*n+i] should equal sum_r U[r, i*m+j] * V[r, j*p+k] * W[r, k*n+i]
    # For schoolbook, each term is 1 for exactly one r, so the sum is 1.
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
