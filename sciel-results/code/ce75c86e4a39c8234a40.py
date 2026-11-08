import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 110

    n, m, p = 2, 4, 8
    
    # Rank 56 decomposition using Strassen per 2x2 sub-block
    x = np.array([
        [1, 0, 0, 1],
        [0, 0, 1, 1],
        [1, 0, 0, 0],
        [0, 0, 0, 1],
        [1, 1, 0, 0],
        [-1, 0, 1, 0],
        [0, 1, 0, -1]
    ], dtype=int)
    
    y = np.array([
        [1, 0, 0, 1],
        [1, 0, 0, 0],
        [0, 1, 0, -1],
        [-1, 0, 1, 0],
        [0, 0, 0, 1],
        [1, 1, 0, 0],
        [0, 0, 1, 1]
    ], dtype=int)
    
    z = np.array([
        [1, 0, 0, 1, -1, 0, 1],
        [0, 0, 1, 0, 1, 0, 0],
        [0, 1, 0, 1, 0, 0, 0],
        [1, -1, 1, 0, 0, 1, 0]
    ], dtype=int)
    
    rank = 56
    U = np.zeros((n*m, rank), dtype=float)
    V = np.zeros((m*p, rank), dtype=float)
    W = np.zeros((p*n, rank), dtype=float)
    
    r = 0
    for ab in range(2):
        for bg in range(4):
            for s in range(7):
                for i in range(n):
                    for jp in range(2):
                        U[i*m + ab*2 + jp, r] = x[s, i*2 + jp]
                for jp in range(2):
                    for kp in range(2):
                        V[(ab*2+jp)*p + 2*bg + kp, r] = y[s, jp*2 + kp]
                for i in range(n):
                    for kp in range(2):
                        W[(2*bg+kp)*n + i, r] = z[i*2+kp, s]
                r += 1
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
