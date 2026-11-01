import numpy as np

DIMENSIONS = [2, 4, 6]

def construct():
    n, m, p = DIMENSIONS
    # Rank is 24
    rank = 24
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Block 1: A[:, 0:2] * B[0:2, :]
    # A indices: i in 0..1, j in 0..1 -> U index i*m + j
    # B indices: j in 0..1, k in 0..5 -> V index j*p + k
    # C indices: i in 0..1, k in 0..5 -> W index k*n + i
    
    col = 0
    for i in range(n):
        for j in range(2): # j is 0, 1
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
                
    # Block 2: A[:, 2:4] * B[2:4, :]
    # A indices: i in 0..1, j in 2..3 -> U index i*m + j
    # B indices: j in 2..3, k in 0..5 -> V index j*p + k
    # C indices: i in 0..1, k in 0..5 -> W index k*n + i
    
    for i in range(n):
        for j in range(2, m): # j is 2, 3
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
                
    return {"U": U, "V": V, "W": W}
