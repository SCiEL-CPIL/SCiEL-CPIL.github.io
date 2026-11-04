import numpy as np

DIMENSIONS = [2, 5, 5]

def construct():
    n, m, p = DIMENSIONS
    rank = n * m * p
    
    U = np.zeros((n * m, rank), dtype=int)
    V = np.zeros((m * p, rank), dtype=int)
    W = np.zeros((p * n, rank), dtype=int)
    
    # Trivial decomposition
    for r in range(rank):
        # Map rank index r to (i, j, k)
        # r = i * m * p + j * p + k
        i = r // (m * p)
        rem = r % (m * p)
        j = rem // p
        k = rem % p
        
        U[i * m + j, r] = 1
        V[j * p + k, r] = 1
        W[k * n + i, r] = 1
        
    # Convert to lists to match interface expectation if strict
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
