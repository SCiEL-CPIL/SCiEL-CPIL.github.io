import numpy as np

DIMENSIONS = [2, 2, 12]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank = 1

    # The matrix multiplication tensor T[i, j, k] = 1 for all i in [0, n), j in [0, m), k in [0, p)
    # This is a rank-1 tensor: T[i, j, k] = 1 * 1 * 1
    # U shape: (n*m, rank)
    # V shape: (m*p, rank)
    # W shape: (p*n, rank)
    
    U = np.ones((n * m, rank), dtype=int)
    V = np.ones((m * p, rank), dtype=int)
    W = np.ones((p * n, rank), dtype=int)
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
