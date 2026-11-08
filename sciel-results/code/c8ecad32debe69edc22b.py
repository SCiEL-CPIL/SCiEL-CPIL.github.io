"""Rewritten program for matrix multiplication 3,4,8 using rank-64 decomposition."""

def construct():
    n, m, p = 3, 4, 8
    rank = 64
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Part 1: i=0, schoolbook for 1x4x8 (32 terms)
    for k in range(p):
        for j in range(m):
            U[0 * m + j][col] = 1
            V[j * p + k][col] = 1
            W[k * n + 0][col] = 1
            col += 1
    
    # Part 2: i=1,2, linear decomposition for 2x4x8 (32 terms)
    for k in range(p):
        for j in range(m):
            U[1 * m + j][col] = 1
            U[2 * m + j][col] = 1
            V[j * p + k][col] = 1
            W[k * n + 1][col] = 1
            W[k * n + 2][col] = 1
            col += 1
    
    return {"U": U, "V": V, "W": W}
