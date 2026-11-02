import time
import numpy as np

DIMENSIONS = [2, 3, 4]

def construct():
    n, m, p = DIMENSIONS
    rank = n * m * p
    
    # Initialize matrices with zeros
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    # Schoolbook decomposition
    # Each term r corresponds to a unique (i, j, k) triplet?
    # No, the sum is over j for each (i, k).
    # The tensor entry T[i, j, k] in the code's convention:
    # U[i, j, r] V[j, k, r] W[k, i, r]
    # We want this sum to equal the target tensor.
    # If the target is the matrix multiplication structure,
    # the term A[i, j] B[j, k] contributes to C[i, k].
    # So for each j, we have a term.
    # There are m terms for each (i, k).
    # Total terms = n * p * m = 24.
    
    col = 0
    for i in range(n):
        for k in range(p):
            for j in range(m):
                # U index: i, j
                U[i * m + j, col] = 1.0
                # V index: j, k
                V[j * p + k, col] = 1.0
                # W index: k, i
                W[k * n + i, col] = 1.0
                col += 1
                
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
