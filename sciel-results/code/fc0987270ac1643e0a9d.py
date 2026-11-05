import numpy as np
import time

DIMENSIONS = [3, 3, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank = n * m * p
    
    # Initialize U, V, W as numpy arrays of zeros
    U = np.zeros((n * m, rank), dtype=int)
    V = np.zeros((m * p, rank), dtype=int)
    W = np.zeros((p * n, rank), dtype=int)
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # U[i*m+j, col] = 1
                U[i * m + j, col] = 1
                # V[j*p+k, col] = 1
                V[j * p + k, col] = 1
                # W[k*n+i, col] = 1
                W[k * n + i, col] = 1
                col += 1
                
    # Convert to lists of lists as per interface requirement
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
# EVOLVE-BLOCK-END
