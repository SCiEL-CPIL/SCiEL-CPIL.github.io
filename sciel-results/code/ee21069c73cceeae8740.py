import numpy as np
import time

DIMENSIONS = [5, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    """
    Constructs the schoolbook decomposition for 5x5 matrix multiplication.
    Rank = 5 * 5 * 5 = 125.
    """
    n, m, p = DIMENSIONS
    rank = n * m * p
    
    # Initialize matrices with zeros
    U = np.zeros((n * m, rank), dtype=np.float64)
    V = np.zeros((m * p, rank), dtype=np.float64)
    W = np.zeros((p * n, rank), dtype=np.float64)
    
    # Fill the matrices according to the schoolbook algorithm
    # U[i*m + j, c] = 1
    # V[j*p + k, c] = 1
    # W[k*n + i, c] = 1
    # where c = i*m*p + j*p + k
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                c = (i * m + j) * p + k
                U[i * m + j, c] = 1.0
                V[j * p + k, c] = 1.0
                W[k * n + i, c] = 1.0
                
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
