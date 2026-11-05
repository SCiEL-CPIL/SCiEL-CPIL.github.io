import numpy as np

DIMENSIONS = [4, 4, 5]

def construct():
    n, m, p = DIMENSIONS
    # We split the problem into 4x4x4 and 4x4x1
    # Rank for 4x4x4 schoolbook is 64
    # Rank for 4x4x1 schoolbook is 4
    # Total rank 68
    
    rank = 68
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Part 1: 4x4x4 (k=0..3)
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(4): # Only first 4 columns
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
                
    # Part 2: 4x4x1 (k=4)
    for i in range(n):
        for j in range(m):
            U[i * m + j][col] = 1
            V[j * p + 4][col] = 1
            W[4 * n + i][col] = 1
            col += 1
            
    return {"U": U, "V": V, "W": W}
