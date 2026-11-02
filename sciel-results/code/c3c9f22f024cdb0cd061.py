import time
import numpy as np

DIMENSIONS = [3, 5, 7]

def construct():
    n, m, p = DIMENSIONS
    rank = n * m  # 3 * 5 = 15
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for i in range(n):
        for j in range(m):
            t = i * m + j
            U[i * m + j][t] = 1
            for k in range(p):
                V[j * p + k][t] = 1
                W[k * n + i][t] = 1
    
    return {"U": U, "V": V, "W": W}
