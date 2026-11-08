import numpy as np
import time

DIMENSIONS = [3, 5, 6]

def construct():
    n, m, p = DIMENSIONS
    rank = n * m  # 3*5 = 15
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    for i in range(n):
        for k in range(m):
            r = i * m + k
            U[i * m + k, r] = 1
            for j in range(p):
                V[k * p + j, r] = 1
                W[j * n + i, r] = 1
                
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
