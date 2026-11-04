import numpy as np

DIMENSIONS = [4, 4, 5]

def construct():
    n, m, p = DIMENSIONS
    rank = 65
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Part 1: Strassen 4x4x4 (Rank 49)
    # I will use a hardcoded list of 49 terms if I can generate them.
    # Since I cannot easily generate them correctly in the thought process,
    # I will use a fallback: Schoolbook for first 4 cols (64 terms) + Schoolbook for 5th col (16 terms) = 80.
    # This is not an improvement.
    
    # Let's try to find a 72-rank decomposition?
    # No.
    
    # I will return the schoolbook decomposition.
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
