import numpy as np

DIMENSIONS = [2, 3, 3]

# EVOLVE-BLOCK-START
def construct():
    # Approach: Tensor unfolding and basis extraction
    # This constructs the trivial decomposition by reshaping identity matrices.
    n, m, p = DIMENSIONS
    rank = n * m * p
    
    # The multiplication tensor T is defined by T[i, j, k] = 1 if ...
    # The trivial decomposition corresponds to the 15 (actually 18) basis tensors.
    
    # Create U, V, W
    U = np.zeros((n * m, rank), dtype=int)
    V = np.zeros((m * p, rank), dtype=int)
    W = np.zeros((p * n, rank), dtype=int)
    
    # Map (i, j, k) to index r
    # r = i*m*p + j*p + k
    for i in range(n):
        for j in range(m):
            for k in range(p):
                r = i * m * p + j * p + k
                U[i * m + j, r] = 1
                V[j * p + k, r] = 1
                W[k * n + i, r] = 1
                
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
