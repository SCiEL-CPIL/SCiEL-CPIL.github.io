"""Tensor decomposition search for 3x5x5 matrix multiplication."""
DIMENSIONS = [3, 5, 5]

# EVOLVE-BLOCK-START
import numpy as np

def construct():
    """
    Returns a schoolbook tensor decomposition for 3x5x5 matrix multiplication.
    U: (n*m, rank), V: (m*p, rank), W: (p*n, rank)
    The decomposition satisfies: for all i,j,k,
        sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1
    """
    n, m, p = DIMENSIONS
    rank = n * m * p
    
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    c = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, c] = 1.0
                V[j * p + k, c] = 1.0
                W[k * n + i, c] = 1.0
                c += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
