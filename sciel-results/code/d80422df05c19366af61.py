import time
import numpy as np

def construct():
    """
    Returns a low-rank tensor decomposition for 4x4 * 4x9 matrix multiplication.
    Uses schoolbook decomposition.
    U: (4*4, rank) = (16, rank)
    V: (4*9, rank) = (36, rank)
    W: (9*4, rank) = (36, rank)
    """
    n, m, p = 4, 4, 9
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # U index: i*m + j
                # V index: j*p + k
                # W index: k*n + i
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
                
    return {"U": U, "V": V, "W": W}
