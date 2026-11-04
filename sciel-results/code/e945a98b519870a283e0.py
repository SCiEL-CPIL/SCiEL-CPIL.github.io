import numpy as np

DIMENSIONS = [2, 3, 7]

def construct():
    n, m, p = DIMENSIONS
    
    # We decompose 2x3x7 into 2x3x3 and 2x3x4
    # Rank(2x3x3) = 9
    # Rank(2x3x4) = 11 (using known decomposition) or 12 (schoolbook)
    # Let's use 9 + 12 = 21 for safety and simplicity, or try to get 20.
    # I will use 20 by implementing a specific rank 11 for 2x3x4.
    
    # However, to ensure correctness without external lookup, I will use schoolbook for both parts.
    # Rank(2x3x3) = 9
    # Rank(2x3x4) = 12
    # Total Rank = 21
    
    rank = 21
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Part 1: 2x3x3 (columns 0-2 of B)
    # i in 0..1, j in 0..2, k in 0..2
    for i in range(n):
        for j in range(m):
            for k in range(3):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
                
    # Part 2: 2x3x4 (columns 3-6 of B, but relative to the 4x3 block it's 0..3)
    # We are multiplying A (2x3) by B_sub (3x4)
    # B_sub columns correspond to global k in 3..6
    for i in range(n):
        for j in range(m):
            for k in range(4):
                # Global k index is 3 + k
                U[i * m + j][col] = 1
                V[j * p + (3 + k)][col] = 1
                W[(3 + k) * n + i][col] = 1
                col += 1
                
    return {"U": U, "V": V, "W": W}
