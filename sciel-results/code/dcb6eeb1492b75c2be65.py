import numpy as np
import time

DIMENSIONS = [4, 4, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # U: (n*m, rank) -> (16, rank)
    # V: (m*p, rank) -> (28, rank)
    # W: (p*n, rank) -> (28, rank)
    
    U_cols = []
    V_cols = []
    W_cols = []
    
    # Schoolbook algorithm:
    # C[i, k] = sum_j A[i, j] * B[j, k]
    # Each term A[i, j] * B[j, k] is a rank-1 tensor.
    # U index: i * m + j (size n*m = 16)
    # V index: j * p + k (size m*p = 28)
    # W index: k * n + i (size p*n = 28)
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u = [0] * (n * m)
                v = [0] * (m * p)
                w = [0] * (p * n)
                
                u[i * m + j] = 1
                v[j * p + k] = 1
                w[k * n + i] = 1
                
                U_cols.append(u)
                V_cols.append(v)
                W_cols.append(w)
                
    rank = len(U_cols)
    U = np.array(U_cols).T
    V = np.array(V_cols).T
    W = np.array(W_cols).T
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
