import numpy as np

DIMENSIONS = [2, 5, 5]

def construct():
    n, m, p = DIMENSIONS
    rank = n * m * p
    
    # Construct U, V, W using vectorized operations or different logic
    # U shape: (n*m, rank)
    # V shape: (m*p, rank)
    # W shape: (p*n, rank)
    
    # The trivial decomposition assigns a unique rank term to each (i, j, k) triplet.
    # U[i*m + j, r] = 1 if r corresponds to (i, j, k)
    # V[j*p + k, r] = 1 if r corresponds to (i, j, k)
    # W[k*n + i, r] = 1 if r corresponds to (i, j, k)
    
    # We can generate these using meshgrid or simple indexing.
    # r index = i * m * p + j * p + k
    
    U = np.zeros((n * m, rank), dtype=int)
    V = np.zeros((m * p, rank), dtype=int)
    W = np.zeros((p * n, rank), dtype=int)
    
    for r in range(rank):
        # Decode r into i, j, k
        # r = i * (m * p) + j * p + k
        k = r % p
        j = (r // p) % m
        i = (r // (m * p)) % n
        
        U[i * m + j, r] = 1
        V[j * p + k, r] = 1
        W[k * n + i, r] = 1
        
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
