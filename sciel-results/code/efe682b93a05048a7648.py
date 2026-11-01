import time
import numpy as np

DIMENSIONS = [2, 2, 13]

def construct():
    start_time = time.time()
    deadline = start_time + 110
    
    n, m, p = DIMENSIONS
    nm, mp, pn = n * m, m * p, p * n
    
    # Schoolbook decomposition
    R = n * m * p
    U = np.zeros((nm, R), dtype=int)
    V = np.zeros((mp, R), dtype=int)
    W = np.zeros((pn, R), dtype=int)
    
    col = 0
    for ii in range(n):
        for jj in range(m):
            for kk in range(p):
                U[ii*m+jj, col] = 1
                V[jj*p+kk, col] = 1
                W[kk*n+ii, col] = 1
                col += 1
                
    return {"U": U, "V": V, "W": W}
