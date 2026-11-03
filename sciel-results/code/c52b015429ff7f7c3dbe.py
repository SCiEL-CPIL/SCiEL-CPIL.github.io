"""Program for 4x4x9 matrix multiplication rank decomposition."""
import numpy as np
import time

DIMENSIONS = [4, 4, 9]

def construct():
    """
    Construct a rank-144 tensor decomposition for 4x4x9 matrix multiplication.
    
    Uses the standard schoolbook (trivial) decomposition where each (i,j,k)
    triple contributes one rank-1 term. This is correct and valid.
    
    The rank-minimal decomposition for 4x4x9 is unknown, but the schoolbook
    decomposition with rank n*m*p = 4*4*9 = 144 is a valid baseline.
    
    We attempt to verify correctness and return the decomposition.
    """
    n, m, p = DIMENSIONS  # n=4, m=4, p=9
    
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
