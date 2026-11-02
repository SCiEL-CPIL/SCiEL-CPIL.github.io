import numpy as np

DIMENSIONS = [2, 4, 4]

def construct():
    n, m, p = DIMENSIONS
    rank = n * m * p
    
    # Create identity-like structures for the bilinear form
    # U is n*m x rank
    # V is m*p x rank
    # W is p*n x rank
    
    # We iterate over the "core" indices i, j, k
    # For each (i, j, k), we have a rank-1 term.
    # U[i*m + j, t] = 1
    # V[j*p + k, t] = 1
    # W[k*n + i, t] = 1
    
    U = np.zeros((n * m, rank), dtype=int)
    V = np.zeros((m * p, rank), dtype=int)
    W = np.zeros((p * n, rank), dtype=int)
    
    t = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                
                U[u_idx, t] = 1
                V[v_idx, t] = 1
                W[w_idx, t] = 1
                t += 1
                
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
