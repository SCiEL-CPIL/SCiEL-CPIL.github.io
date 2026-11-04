import time
import sys

DIMENSIONS = [3, 3, 5]

def construct():
    n, m, p = DIMENSIONS
    # We decompose into 3x3 part (first 3 cols of B) and 3x2 part (last 2 cols of B)
    # Rank of 3x3 schoolbook is 27
    # Rank of 3x2 schoolbook is 6
    # Total rank 33
    
    rank = 33
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Part 1: 3x3 multiplication (cols 0, 1, 2 of B)
    # Schoolbook for 3x3
    for i in range(n):
        for j in range(m):
            for k in range(3): # k is 0, 1, 2
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
                
    # Part 2: 3x2 multiplication (cols 3, 4 of B)
    # Schoolbook for 3x2
    for i in range(n):
        for j in range(m):
            for k in range(3, p): # k is 3, 4
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
                
    return {"U": U, "V": V, "W": W}
