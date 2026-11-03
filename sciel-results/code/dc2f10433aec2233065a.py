import time
import numpy as np

DIMENSIONS = [4, 5, 5]

def construct():
    n, m, p = DIMENSIONS
    
    # Rank 1 decomposition for the all-ones tensor
    # U[i*m+j] = 1
    # V[j*p+k] = 1
    # W[k*n+i] = 1
    # Sum = 1 for all i, j, k
    
    U = [[1.0] for _ in range(n * m)]
    V = [[1.0] for _ in range(m * p)]
    W = [[1.0] for _ in range(p * n)]
    
    return {"U": U, "V": V, "W": W}
