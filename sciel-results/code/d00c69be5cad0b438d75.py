import numpy as np

def construct():
    n, m, p = 3, 5, 5
    rank = n * m * p  # 75
    
    U = np.zeros((n * m, rank), dtype=int)
    V = np.zeros((m * p, rank), dtype=int)
    W = np.zeros((p * n, rank), dtype=int)
    
    # Natural (trivial) tensor decomposition
    # U has shape (n*m, rank), indexed by (i, j)
    # V has shape (m*p, rank), indexed by (j, k)
    # W has shape (p*n, rank), indexed by (k, i)  # trace convention: W[k*n + i, r]
    
    r = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, r] = 1
                V[j * p + k, r] = 1
                W[k * n + i, r] = 1
                r += 1
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
