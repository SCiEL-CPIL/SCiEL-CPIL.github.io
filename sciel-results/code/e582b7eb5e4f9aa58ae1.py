import numpy as np
import time

_START = time.time()
_DEADLINE = _START + 115.0

def construct():
    n, m, p = 2, 4, 8
    A_dim = n * m
    B_dim = m * p
    C_dim = p * n
    rank = n * m * p
    
    U = np.zeros((A_dim, rank))
    V = np.zeros((B_dim, rank))
    W = np.zeros((C_dim, rank))
    
    for i in range(n):
        for j in range(m):
            for l in range(p):
                k = (i * m + j) * p + l
                a_idx = i * m + j
                b_idx = j * p + l
                # Fixed index calculation for W based on "W[k*n+i,r]" convention
                # where k iterates columns (l) and i is rows
                c_idx = l * n + i
                
                U[a_idx, k] = 1.0
                V[b_idx, k] = 1.0
                W[c_idx, k] = 1.0
                
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
