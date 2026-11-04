import numpy as np
import time

def construct():
    n, m, p = 3, 5, 6
    nm, mp, pn = n * m, m * p, p * n
    
    # Schoolbook fallback (rank 90)
    U = np.zeros((nm, nm * mp))
    V = np.zeros((mp, nm * mp))
    W = np.zeros((pn, nm * mp))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
