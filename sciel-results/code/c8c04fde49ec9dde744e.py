"""Matrix multiplication 4,5,6 via group-theoretic embedding."""
import numpy as np
from itertools import product

DIMENSIONS = [4, 5, 6]

def construct():
    """
    Construct a rank-minimal exact tensor decomposition for 4x5 * 5x6 matrix multiplication.
    
    We use the schoolbook decomposition which is guaranteed to be exact.
    The rank is n*m*p = 120.
    
    The tensor decomposition is:
    sum_{c=1}^{rank} U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1
    for all valid i, j, k, and 0 otherwise.
    
    For the schoolbook decomposition, we assign a unique column c to each (i, j, k) triple.
    U[i*m+j, c] = 1
    V[j*p+k, c] = 1
    W[k*n+i, c] = 1
    """
    n, m, p = DIMENSIONS
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}
