import numpy as np

DIMENSIONS = [4, 4, 8]

def construct():
    n, m, p = DIMENSIONS
    # The matrix multiplication tensor T[i, j, k] = 1 for all i, j, k
    # because C[i, k] = sum_j A[i, j] B[j, k]
    # This tensor is the all-ones tensor, which has rank 1.
    # U, V, W are vectors of ones.
    
    rank = 1
    U = np.ones((n * m, rank))
    V = np.ones((m * p, rank))
    W = np.ones((p * n, rank))
    
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
