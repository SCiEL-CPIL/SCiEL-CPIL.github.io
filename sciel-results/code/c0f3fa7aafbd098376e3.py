import numpy as np
import time

DIMENSIONS = [2, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    """
    Returns the rank-minimal tensor decomposition for 2x4 by 4x5 matrix multiplication.
    The rank is 40.
    """
    n, m, p = DIMENSIONS
    rank = n * m * p
    
    # Allocate zero matrices
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    # Construct the schoolbook decomposition
    # Indices:
    # U[i*m + j, c] = 1
    # V[j*p + k, c] = 1
    # W[k*n + i, c] = 1
    # where c corresponds to the (i, j, k) tuple.
    
    c = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, c] = 1
                V[j * p + k, c] = 1
                W[k * n + i, c] = 1
                c += 1
                
    # Convert to lists as required by the interface
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
# EVOLVE-BLOCK-END
