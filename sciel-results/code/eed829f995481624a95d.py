import time
import sys

DIMENSIONS = [3, 4, 6]

def construct():
    n, m, p = DIMENSIONS
    # We decompose the 3x4 * 4x6 multiplication into two 3x3 * 4x3 multiplications?
    # No, we split the 6 columns of B into two groups of 3.
    # B = [B1, B2] where B1 is 4x3 and B2 is 4x3.
    # C = [C1, C2] where C1 = A*B1 (3x3) and C2 = A*B2 (3x3).
    # We use schoolbook for each 3x3 multiplication.
    # Rank of 3x3x3 is 27.
    # Total rank = 27 + 27 = 54.
    
    rank = 54
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    # Block 1: C1 = A * B1
    # A is 3x4, B1 is 4x3 (cols 0-2 of B), C1 is 3x3 (cols 0-2 of C)
    for i in range(n):
        for j in range(m):
            for k in range(3):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
                
    # Block 2: C2 = A * B2
    # A is 3x4, B2 is 4x3 (cols 3-5 of B), C2 is 3x3 (cols 3-5 of C)
    for i in range(n):
        for j in range(m):
            for k_local in range(3):
                k = k_local + 3
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
                
    return {"U": U, "V": V, "W": W}
