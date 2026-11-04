import time
import numpy as np

DIMENSIONS = [2, 3, 6]

def _schoolbook():
    """
    Returns the schoolbook tensor decomposition for 2x3x6 matrix multiplication.
    Rank = 36.
    """
    n, m, p = DIMENSIONS
    rank = n * m * p
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1.0
                V[j * p + k][column] = 1.0
                W[k * n + i][column] = 1.0
                column += 1
                
    return {"U": U, "V": V, "W": W}

def construct():
    """
    Constructs the tensor decomposition.
    Currently returns the schoolbook solution (Rank 36).
    """
    return _schoolbook()
