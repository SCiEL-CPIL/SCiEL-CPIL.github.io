import numpy as np
import time

DIMENSIONS = [3, 3, 4]

def construct():
    n, m, p = DIMENSIONS
    # Rank 12 solution
    R = n * p # 3 * 4 = 12
    
    U = [[0] * R for _ in range(n * m)]
    V = [[0] * R for _ in range(m * p)]
    W = [[0] * R for _ in range(p * n)]
    
    # r maps to (i, k)
    # i in 0..n-1, k in 0..p-1
    r = 0
    for i in range(n):
        for k in range(p):
            # W[k*n + i, r] = 1
            W[k * n + i][r] = 1
            
            # U[i*m + j, r] = 1 for all j
            for j in range(m):
                U[i * m + j][r] = 1
                
            # V[j*p + k, r] = 1 for all j
            for j in range(m):
                V[j * p + k][r] = 1
                
            r += 1
            
    return {"U": U, "V": V, "W": W}
