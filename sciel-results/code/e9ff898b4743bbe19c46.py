"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

DIMENSIONS = [2, 5, 5]

def construct():
    n, m, p = DIMENSIONS
    
    # The tensor for matrix multiplication A (n x m) * B (m x p) = C (n x p)
    # is defined by T[i, j, k] = 1 for all i in [0, n), j in [0, m), k in [0, p).
    # This is because C[i, k] = sum_j A[i, j] * B[j, k].
    # The term A[i, j] * B[j, k] contributes to C[i, k] with coefficient 1.
    
    # The rank of the all-ones tensor is 1.
    # We can decompose it using a single rank-1 component.
    # U[i*m + j, 0] = 1
    # V[j*p + k, 0] = 1
    # W[k*n + i, 0] = 1
    
    # Then sum_c U[i,j,c] * V[j,k,c] * W[k,i,c] = 1 * 1 * 1 = 1.
    
    rank = 1
    U = np.ones((n * m, rank))
    V = np.ones((m * p, rank))
    W = np.ones((p * n, rank))
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
