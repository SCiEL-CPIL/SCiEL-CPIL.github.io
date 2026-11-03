"""Program for 3x4x5 matrix multiplication tensor decomposition."""
import numpy as np


def construct():
    n, m, p = 3, 4, 5
    
    # Schoolbook decomposition: rank = n*m*p = 60
    # U has shape (n*m, rank) = (12, 60)
    # V has shape (m*p, rank) = (20, 60)  
    # W has shape (p*n, rank) = (15, 60)
    # W follows trace convention W[k*n + i, r]
    
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # U index: i*m + j
                U[i * m + j][col] = 1
                # V index: j*p + k
                V[j * p + k][col] = 1
                # W index: k*n + i (trace convention)
                W[k * n + i][col] = 1
                col += 1
    
    return {"U": U, "V": V, "W": W}
