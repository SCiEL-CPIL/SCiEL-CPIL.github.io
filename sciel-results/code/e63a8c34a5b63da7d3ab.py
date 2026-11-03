"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
import random

DIMENSIONS = [3, 3, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # T[a, b, c] where a=i*m+j, b=j*p+k, c=k*n+i
    # T[i*m+j, j*p+k, k*n+i] = 1
    # U: (n*m, rank), V: (m*p, rank), W: (p*n, rank)
    # T[a,b,c] = sum_r U[a,r]*V[b,r]*W[c,r]
    
    nm = n * m  # 9
    mp = m * p  # 18
    pn = p * n  # 18
    
    # Return the schoolbook decomposition immediately to ensure completion within time limit.
    # Rank = n*m*p = 54
    rank = nm * mp * pn
    U = np.zeros((nm, rank))
    V = np.zeros((mp, rank))
    W = np.zeros((pn, rank))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                U[a, col] = 1
                V[b, col] = 1
                W[c, col] = 1
                col += 1
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
