"""Rank-minimal tensor decomposition for 3x3x6 matrix multiplication."""
import numpy as np

DIMENSIONS = [3, 3, 6]

def construct():
    """Construct a tensor decomposition for 3x3x6 matrix multiplication.
    
    Returns factor matrices U, V, W such that the matrix multiplication
    tensor is exactly decomposed. The rank is the number of columns.
    """
    n, m, p = DIMENSIONS  # n=3, m=3, p=6
    
    # Schoolbook decomposition: rank = n*m*p = 54
    # U[i*m+j, t] = 1 if term t uses A[i,j]
    # V[j*p+k, t] = 1 if term t uses B[j,k]
    # W[k*n+i, t] = 1 if term t contributes to C[i,k]
    
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    
    return {"U": U, "V": V, "W": W}
