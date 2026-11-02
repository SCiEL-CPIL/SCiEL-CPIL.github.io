import numpy as np
import time
from itertools import product

DIMENSIONS = [3, 3, 7]

def find_3x3x4_decomposition(max_rank=36, time_limit=60):
    """Find a low-rank decomposition for the 3x3x4 all-ones tensor."""
    n, m, p = 3, 3, 4
    start_time = time.time()
    
    # Schoolbook
    U = np.zeros((n, max_rank))
    V = np.zeros((m, max_rank))
    W = np.zeros((p, max_rank))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i, col] = 1
                V[j, col] = 1
                W[k, col] = 1
                col += 1
    
    # Try to reduce rank
    # This is a simplified search
    for r in range(20, max_rank + 1):
        if time.time() - start_time > time_limit:
            break
        # Try to find a rank-r decomposition
        # This is a placeholder; a real search would use ILP or optimization
        pass
    
    return U[:n, :max_rank], V[:m, :max_rank], W[:p, :max_rank]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140
    
    # Schoolbook decomposition
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
    
    # Return schoolbook for now
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
