import numpy as np

DIMENSIONS = [2, 3, 7]

def construct():
    n, m, p = 2, 3, 7
    U = np.zeros((n * m, 42), dtype=np.float64)
    V = np.zeros((m * p, 42), dtype=np.float64)
    W = np.zeros((p * n, 42), dtype=np.float64)
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
