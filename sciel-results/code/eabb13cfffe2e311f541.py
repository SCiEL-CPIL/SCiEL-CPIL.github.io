import numpy as np

DIMENSIONS = [3, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank = n * m * p
    
    # We construct the factors using the standard schoolbook algorithm.
    # For 3x4x4, the rank is 48.
    # We use a direct loop for clarity and to ensure integer types are correct.
    
    U = np.zeros((n * m, rank), dtype=int)
    V = np.zeros((m * p, rank), dtype=int)
    W = np.zeros((p * n, rank), dtype=int)
    
    # Iterate over all (i, j, k) triples
    # i in 0..n-1, j in 0..m-1, k in 0..p-1
    c = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_row = i * m + j
                v_row = j * p + k
                w_row = k * n + i
                
                U[u_row, c] = 1
                V[v_row, c] = 1
                W[w_row, c] = 1
                c += 1
                
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
