import numpy as np

DIMENSIONS = [2, 3, 9]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    r = n * m * p
    U = np.zeros((n * m, r), dtype=float)
    V = np.zeros((m * p, r), dtype=float)
    W = np.zeros((p * n, r), dtype=float)
    
    for t in range(r):
        i = t // (m * p)
        rem = t % (m * p)
        j = rem // p
        k = rem % p
        U[i * m + j, t] = 1.0
        V[j * p + k, t] = 1.0
        W[k * n + i, t] = 1.0
        
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
# EVOLVE-BLOCK-END
