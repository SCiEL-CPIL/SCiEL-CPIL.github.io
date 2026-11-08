import numpy as np
import time

def construct():
    n, m, p = 3, 4, 4
    rank = n * m * p
    
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    # Generate the trivial (schoolbook) decomposition
    # The tensor T is 1 at (u, v, w) where:
    # u = i*m + j
    # v = j*p + k
    # w = k*n + i
    # and 0 elsewhere.
    
    c = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                
                U[u_idx, c] = 1.0
                V[v_idx, c] = 1.0
                W[w_idx, c] = 1.0
                c += 1
                
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
