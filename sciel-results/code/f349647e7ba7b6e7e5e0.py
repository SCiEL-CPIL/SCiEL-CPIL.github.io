import numpy as np

DIMENSIONS = [3, 3, 7]

def construct():
    n, m, p = DIMENSIONS
    # We use the additivity of rank for disjoint output/input blocks.
    # M_{3,3,7} = M_{3,3,3} (cols 0-2) + M_{3,3,3} (cols 3-5) + M_{3,3,1} (col 6)
    # Rank = R(M_{3,3,3}) + R(M_{3,3,3}) + R(M_{3,3,1})
    # We use schoolbook for M_{3,3,3} (rank 9) and M_{3,3,1} (rank 3) for simplicity and correctness.
    # Total rank = 9 + 9 + 3 = 21.
    
    rank = 21
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    # Block 1: cols 0-2 of B and C
    for i in range(n):
        for j in range(m):
            for k in range(3):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
                
    # Block 2: cols 3-5 of B and C
    for i in range(n):
        for j in range(m):
            for k in range(3, 6):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1

    # Block 3: col 6 of B and C
    for i in range(n):
        for j in range(m):
            k = 6
            U[i * m + j][col] = 1
            V[j * p + k][col] = 1
            W[k * n + i][col] = 1
            col += 1

    return {"U": U, "V": V, "W": W}
