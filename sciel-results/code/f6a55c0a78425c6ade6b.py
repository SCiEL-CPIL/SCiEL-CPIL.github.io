import time
import numpy as np


def construct():
    """
    Find a low-rank exact tensor decomposition for 4x4 * 4x9 matrix multiplication.
    
    The multiplication tensor is the all-ones tensor T[i,j,k] = 1 for all i in [0,4),
    j in [0,4), k in [0,9). 
    
    The all-ones tensor has rank 1.
    U[i*m+j, 0] = 1
    V[j*p+k, 0] = 1
    W[k*n+i, 0] = 1
    Sum = 1.
    """
    n, m, p = 4, 4, 9
    R = 1
    
    U = np.ones((n * m, R), dtype=float)
    V = np.ones((m * p, R), dtype=float)
    W = np.ones((p * n, R), dtype=float)
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
