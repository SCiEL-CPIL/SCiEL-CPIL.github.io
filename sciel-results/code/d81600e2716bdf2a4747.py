import numpy as np

DIMENSIONS = [3, 4, 6]

def construct():
    n, m, p = DIMENSIONS
    nm, mp, pn = n*m, m*p, p*n
    rank = n * m * p  # 72
    
    # Initialize matrices
    U = [[0.0] * rank for _ in range(nm)]
    V = [[0.0] * rank for _ in range(mp)]
    W = [[0.0] * rank for _ in range(pn)]
    
    # Fill in the schoolbook decomposition
    # The tensor is T[i, j, k] = 1 if the index corresponds to the term A[i,j] * B[j,k]
    # U[i*m + j, r] = 1
    # V[j*p + k, r] = 1
    # W[k*n + i, r] = 1
    # where r is the index of the (i, j, k) triplet
    
    r = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m + j][r] = 1.0
                V[j*p + k][r] = 1.0
                W[k*n + i][r] = 1.0
                r += 1
                
    return {"U": U, "V": V, "W": W}
