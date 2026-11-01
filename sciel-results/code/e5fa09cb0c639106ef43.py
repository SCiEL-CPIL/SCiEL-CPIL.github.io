import numpy as np

DIMENSIONS = [2, 3, 9]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    
    # The rank is n*m*p for this specific structure because the output dimensions
    # are disjoint for each column k of B, making the tensor a direct sum of
    # p independent tensors of rank n*m.
    rank = n * m * p
    
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    r = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # Index mapping based on trace convention W[k*n+i, r]
                # U corresponds to A[i, j] -> index i*m + j
                U[i * m + j, r] = 1.0
                # V corresponds to B[j, k] -> index j*p + k
                V[j * p + k, r] = 1.0
                # W corresponds to C[i, k] -> index k*n + i
                W[k * n + i, r] = 1.0
                r += 1
                
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
# EVOLVE-BLOCK-END
