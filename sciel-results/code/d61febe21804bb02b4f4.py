import numpy as np
import time

DIMENSIONS = [3, 5, 6]

def construct():
    n, m, p = DIMENSIONS
    # Target Tensor T of shape (n*m, m*p, p*n)
    # T[I, J, K] = 1 if indices are consistent
    # I = i*m + j
    # J = j*p + k
    # K = k*n + i
    
    # Precompute consistent indices
    # I ranges 0..n*m-1
    # J ranges 0..m*p-1
    # K ranges 0..p*n-1
    
    # Dense is 15*30*18 = 8100 floats. Very small.
    T = np.zeros((n*m, m*p, p*n), dtype=np.float64)
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                I = i * m + j
                J = j * p + k
                K = k * n + i
                T[I, J, K] = 1.0

    # Fallback to schoolbook
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
