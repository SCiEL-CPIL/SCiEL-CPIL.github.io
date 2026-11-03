import numpy as np
import time

DIMENSIONS = [4, 4, 5]

def construct():
    n, m, p = DIMENSIONS
    
    # Schoolbook for 4x4x4 part (first 4 columns of B)
    # Rank 64
    # Plus schoolbook for 4x4x1 part (last 1 column of B)
    # Rank 4
    # Total Rank 68
    
    rank = 64 + 4
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Part 1: 4x4x4 (columns 0-3 of B)
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(4): # p=4 for this part
                U[i * m + j][col] = 1
                V[j * 4 + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
                
    # Part 2: 4x4x1 (column 4 of B)
    # Note: V index is j*p + k. Here p=5. So j*5 + 4.
    # W index is k*n + i. Here k=4. So 4*n + i.
    for i in range(n):
        for j in range(m):
            k = 4
            U[i * m + j][col] = 1
            V[j * p + k][col] = 1
            W[k * n + i][col] = 1
            col += 1
            
    return {"U": U, "V": V, "W": W}
