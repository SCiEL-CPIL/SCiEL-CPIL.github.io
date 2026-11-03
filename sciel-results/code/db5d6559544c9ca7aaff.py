import time
import numpy as np

DIMENSIONS = [2, 4, 4]

def construct():
    n, m, p = DIMENSIONS
    rank = m  # which is 4
    
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    # U[i*m + j, r] = delta_{j, r}
    for i in range(n):
        for j in range(m):
            U[i * m + j, j] = 1.0
            
    # V[j*p + k, r] = delta_{j, r}
    for j in range(m):
        for k in range(p):
            V[j * p + k, j] = 1.0
            
    # W[k*n + i, r] = delta_{k, r}
    for k in range(p):
        for i in range(n):
            W[k * n + i, k] = 1.0
            
    return {
        "U": [[float(x) for x in row] for row in U],
        "V": [[float(x) for x in row] for row in V],
        "W": [[float(x) for x in row] for row in W]
    }
