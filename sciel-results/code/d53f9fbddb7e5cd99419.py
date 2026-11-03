import time
import numpy as np

DIMENSIONS = [2, 4, 8]

def construct():
    start_time = time.time()
    n, m, p = DIMENSIONS  # n=2, m=4, p=8

    rank = n  # 2
    
    U = [[0]*rank for _ in range(n*m)]
    V = [[0]*rank for _ in range(m*p)]
    W = [[0]*rank for _ in range(p*n)]
    
    for i in range(n):
        r = i
        # U[i*m + j, r] = 1 for all j
        for j in range(m):
            U[i*m + j][r] = 1
        # V[j*p + i, r] = 1 for all j
        for j in range(m):
            V[j*p + i][r] = 1
        # W[i*n + i, r] = 1
        W[i*n + i][r] = 1

    if time.time() - start_time > 120:
        pass

    return {"U": U, "V": V, "W": W}
