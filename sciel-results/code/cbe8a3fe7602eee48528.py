import numpy as np

def construct():
    n, m, p = 2, 3, 6
    
    # The condition is:
    # sum_alpha U[i*m+j, alpha] * V[j*p+k, alpha] * W[k*n+i, alpha] = 1 for all i, j, k
    # This is the all-ones tensor of shape (n, m, p).
    # The rank of the all-ones tensor is 1.
    # We can achieve this with U, V, W being all-ones vectors in the rank dimension.
    
    rank = 1
    
    U = np.ones((n * m, rank), dtype=np.int64)
    V = np.ones((m * p, rank), dtype=np.int64)
    W = np.ones((p * n, rank), dtype=np.int64)
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
