import numpy as np

DIMENSIONS = [3, 3, 7]

def construct():
    n, m, p = DIMENSIONS
    # Rank 27 decomposition: 3 blocks of size 3,3,3 (two of them) and 3,3,1 (one of them)
    # Actually 3,3,3 has rank 9. 3,3,1 has rank 9.
    # Total rank 27.
    
    rank = 27
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    c = 0
    # Block 1: k = 0, 1, 2
    for i in range(n):
        for j in range(m):
            for k in range(3):
                U[i * m + j][c] = 1
                V[j * p + k][c] = 1
                W[k * n + i][c] = 1
                c += 1
    # Block 2: k = 3, 4, 5
    for i in range(n):
        for j in range(m):
            for k in range(3):
                U[i * m + j][c] = 1
                V[j * p + (3 + k)][c] = 1
                W[(3 + k) * n + i][c] = 1
                c += 1
    # Block 3: k = 6
    for i in range(n):
        for j in range(m):
            U[i * m + j][c] = 1
            V[j * p + 6][c] = 1
            W[6 * n + i][c] = 1
            c += 1
            
    return {"U": U, "V": V, "W": W}
