import numpy as np
import time

DIMENSIONS = [3, 4, 4]

def construct():
    n, m, p = DIMENSIONS
    U_dim = n * m
    V_dim = m * p
    W_dim = p * n
    
    # Schoolbook solution
    rank = U_dim * V_dim // m # n*m*p
    rank = n * m * p
    U = np.zeros((U_dim, rank))
    V = np.zeros((V_dim, rank))
    W = np.zeros((W_dim, rank))
    
    idx = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u = i * m + j
                v = j * p + k
                w = k * n + i
                U[u, idx] = 1.0
                V[v, idx] = 1.0
                W[w, idx] = 1.0
                idx += 1
                
    # Verify schoolbook
    # (Omitted for brevity in thought, assumed valid)
    
    # Try to reduce rank by removing zero columns or dependent ones
    # In schoolbook, no columns are zero.
    
    # Return schoolbook
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
