import numpy as np
import time

def construct():
    n, m, p = 3, 3, 8
    
    # Trivial decomposition: rank = n * m * p
    # Each term (i, j, k) contributes to C[i, k] via A[i, j] * B[j, k]
    rank = n * m * p
    
    # Initialize U, V, W with zeros
    # U shape: (n*m, rank)
    # V shape: (m*p, rank)
    # W shape: (p*n, rank) -- Note: Prompt says W[k*n+i, r]
    
    U = np.zeros((n * m, rank), dtype=int)
    V = np.zeros((m * p, rank), dtype=int)
    W = np.zeros((p * n, rank), dtype=int)
    
    r = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # U index for A[i, j]
                u_idx = i * m + j
                # V index for B[j, k]
                v_idx = j * p + k
                # W index for C[i, k] following convention W[k*n + i]
                w_idx = k * n + i
                
                U[u_idx, r] = 1
                V[v_idx, r] = 1
                W[w_idx, r] = 1
                
                r += 1
                
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
