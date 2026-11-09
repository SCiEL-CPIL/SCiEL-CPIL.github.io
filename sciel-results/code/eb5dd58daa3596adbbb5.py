import numpy as np

def construct():
    n, m, p = 3, 4, 5
    rank = n * m * p
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    for r in range(n):
        for k in range(p):
            for j in range(m):
                U[r * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + r][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}
