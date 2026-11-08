"""Program for 3x3x3 matrix multiplication tensor decomposition."""
import time
import numpy as np

DIMENSIONS = [3, 3, 3]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = 110.0  # Leave margin before 120s limit
    
    n, m, p = DIMENSIONS
    # Schoolbook rank = 27
    rank_schoolbook = n * m * p
    
    rank = rank_schoolbook
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, column] = 1
                V[j * p + k, column] = 1
                W[k * n + i, column] = 1
                column += 1
    
    # Return the schoolbook decomposition
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
