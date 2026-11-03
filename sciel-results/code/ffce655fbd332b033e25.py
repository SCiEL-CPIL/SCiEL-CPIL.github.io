import numpy as np
import time

DIMENSIONS = [2, 3, 5]

def construct():
    n, m, p = DIMENSIONS
    
    # Schoolbook decomposition (Rank 30)
    # This is always valid and exact.
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1.0
                V[j * p + k, col] = 1.0
                W[k * n + i, col] = 1.0
                col += 1
                
    # Verify
    # T[i,j,k] = sum_t U[i*m+j, t] V[j*p+k, t] W[k*n+i, t]
    # For schoolbook, only one term is non-zero for each (i,j,k), which is 1.
    
    return {"U": U, "V": V, "W": W}
