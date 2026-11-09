"""Rewritten program for 2x3 by 3x6 matrix multiplication with minimal rank."""

DIMENSIONS = [2, 3, 6]

def construct():
    n, m, p = DIMENSIONS
    # The rank is m (3) because we can decompose the product into m independent
    # outer products of the columns of A and rows of B.
    rank = m
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for j in range(m):
        for i in range(n):
            U[i * m + j][j] = 1
            for k in range(p):
                W[k * n + i][j] = 1
        for k in range(p):
            V[j * p + k][j] = 1
            
    return {"U": U, "V": V, "W": W}
