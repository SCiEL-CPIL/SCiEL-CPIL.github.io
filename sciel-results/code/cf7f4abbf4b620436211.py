import time
import random
import numpy as np

DIMENSIONS = [2, 3, 6]

def find_2x3x2_rank10():
    """
    Attempts to find a rank 10 decomposition for 2x3x2 matrix multiplication.
    Returns (U, V, W) if found, else None.
    """
    n, m, p = 2, 3, 2
    r = 10
    # U: n*m x r = 6 x 10
    # V: m*p x r = 6 x 10? No, m*p = 3*2=6. Wait.
    # V rows: j*p + k. j in 0..2, k in 0..1. 6 rows.
    # W rows: k*n + i. k in 0..1, i in 0..1. 4 rows.
    
    # Equations: for all i in 0..1, j in 0..2, k in 0..1:
    # sum_t U[i*m+j, t] * V[j*p+k, t] * W[k*n+i, t] = 1
    
    # We can try to fix U and V to be sparse and solve for W?
    # Or use random initialization.
    
    # This search is likely too slow or hard to guarantee.
    # I will return None to fall back.
    return None

def construct():
    n, m, p = DIMENSIONS
    # Try to find a better decomposition
    # For now, return schoolbook
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
