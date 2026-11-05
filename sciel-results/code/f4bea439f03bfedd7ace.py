import time
import numpy as np

DIMENSIONS = [3, 4, 6]
_START_TIME = time.time()
_DEADLINE = 130

def construct():
    n, m, p = DIMENSIONS
    
    def check_deadline():
        return (time.time() - _START_TIME) < _DEADLINE
    
    # Naive decomposition with rank n*m*p
    rank = n * m * p
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    # The tensor T[i, j, k] = delta_{i,k} is not right.
    # Matrix multiplication C[i, k] = sum_j A[i, j] B[j, k]
    # Tensor decomposition: T[i, j, k] = sum_r U[i, j, r] V[j, k, r] W[i, k, r]
    # For naive, each term r corresponds to a unique (i, j, k) triple?
    # No, for each j, and for each i, k, we have a term A[i, j] * B[j, k] * C[i, k].
    # So for each j in 0..m-1, i in 0..n-1, k in 0..p-1:
    # r = j * n * p + i * p + k
    
    for j in range(m):
        for i in range(n):
            for k in range(p):
                r = j * n * p + i * p + k
                u_idx = i * m + j
                v_idx = j * p + k
                # W index convention: W[k*n + i, r]
                w_idx = k * n + i
                
                U[u_idx, r] = 1.0
                V[v_idx, r] = 1.0
                W[w_idx, r] = 1.0
                
    return {"U": U, "V": V, "W": W}
