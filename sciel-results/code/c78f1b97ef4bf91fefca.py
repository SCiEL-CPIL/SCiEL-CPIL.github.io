"""Matrix multiplication 3x5x5 - rank-minimal exact tensor decomposition."""
import numpy as np

DIMENSIONS = [3, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank = n * m * p
    
    # Build U, V, W using structured index generation
    # U has shape (n*m, rank), V has shape (m*p, rank), W has shape (p*n, rank)
    # Each rank-1 term c corresponds to (i, j, k) with:
    #   U[i*m + j, c] = 1
    #   V[j*p + k, c] = 1
    #   W[k*n + i, c] = 1
    
    U = np.zeros((n * m, rank), dtype=object)
    V = np.zeros((m * p, rank), dtype=object)
    W = np.zeros((p * n, rank), dtype=object)
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                c = i * m * p + j * p + k
                U[i * m + j, c] = 1
                V[j * p + k, c] = 1
                W[k * n + i, c] = 1
    
    # Convert to lists of lists for interface compatibility
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()
    
    # Ensure entries are plain ints (not numpy types)
    U_out = [[int(x) for x in row] for row in U_list]
    V_out = [[int(x) for x in row] for row in V_list]
    W_out = [[int(x) for x in row] for row in W_list]
    
    return {"U": U_out, "V": V_out, "W": W_out}
# EVOLVE-BLOCK-END
