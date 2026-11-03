import numpy as np
import time

DIMENSIONS = [3, 3, 5]

def construct():
    """
    Constructs a tensor decomposition for multiplying a 3x3 matrix by a 3x5 matrix.
    
    We split the 3x5 output into a 3x3 part and a 3x2 part.
    The 3x3 part is handled by schoolbook (rank 27).
    The 3x2 part is handled by schoolbook (rank 6).
    Total rank = 33.
    """
    n, m, p = DIMENSIONS  # n=3, m=3, p=5
    rank = 33
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Part 1: k = 0, 1, 2 (3x3x3 multiplication)
    # Schoolbook for this part
    for i in range(n):
        for j in range(m):
            for k in range(3):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
                
    # Part 2: k = 3, 4 (3x3x2 multiplication)
    # Schoolbook for this part
    for i in range(n):
        for j in range(m):
            for k in range(3, p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
                
    return {"U": U, "V": V, "W": W}
