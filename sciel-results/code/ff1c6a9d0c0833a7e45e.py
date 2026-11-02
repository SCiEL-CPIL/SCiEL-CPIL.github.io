import numpy as np

DIMENSIONS = [2, 3, 6]

def construct():
    n, m, p = DIMENSIONS
    # Rank 18 decomposition using two 2x3x3 schoolbook multiplications
    rank = 18
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    # Part 1: C[:, 0:3] = A @ B[:, 0:3]
    for i in range(n):
        for j in range(m):
            for k in range(3):
                # U index: i*m + j
                # V index: j*p + k  (B is 3x6, so j*p + k)
                # W index: k*n + i
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
                
    # Part 2: C[:, 3:6] = A @ B[:, 3:6]
    for i in range(n):
        for j in range(m):
            for k in range(3):
                # U index: i*m + j
                # V index: j*p + (3 + k)
                # W index: (3 + k)*n + i
                U[i * m + j][col] = 1
                V[j * p + (3 + k)][col] = 1
                W[(3 + k) * n + i][col] = 1
                col += 1
                
    return {"U": U, "V": V, "W": W}
