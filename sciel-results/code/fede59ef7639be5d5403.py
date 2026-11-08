import numpy as np
import time

def construct():
    n, m, p = 4, 4, 5
    # Schoolbook decomposition
    # Rank is n * m * p
    rank = n * m * p
    
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    idx = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # U index: i*m + j
                U[i * m + j, idx] = 1
                # V index: j*p + k
                V[j * p + k, idx] = 1
                # W index: k*n + i (per convention W[k*n+i,r])
                W[k * n + i, idx] = 1
                idx += 1
                
    return {"U": U, "V": V, "W": W}
