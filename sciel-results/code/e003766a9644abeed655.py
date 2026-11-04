import numpy as np

DIMENSIONS = [3, 5, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank = n * m * p
    
    # Initialize lists of lists for U, V, W
    # U shape: (n*m, rank)
    # V shape: (m*p, rank)
    # W shape: (p*n, rank)
    
    U = [[0.0 for _ in range(rank)] for _ in range(n * m)]
    V = [[0.0 for _ in range(rank)] for _ in range(m * p)]
    W = [[0.0 for _ in range(rank)] for _ in range(p * n)]
    
    # Iterate over all (i, j, k) to construct the schoolbook decomposition
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # U index: i*m + j
                u_idx = i * m + j
                # V index: j*p + k
                v_idx = j * p + k
                # W index: k*n + i
                w_idx = k * n + i
                
                U[u_idx][col] = 1.0
                V[v_idx][col] = 1.0
                W[w_idx][col] = 1.0
                col += 1
                
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
