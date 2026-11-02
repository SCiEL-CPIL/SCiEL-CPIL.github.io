"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 5, 7]

import numpy as np

# EVOLVE-BLOCK-START
def construct():
    """
    Returns a rank-minimal exact tensor decomposition for multiplying
    a 3x5 matrix by a 5x7 matrix.
    
    Uses the schoolbook algorithm which corresponds to the standard
    matrix multiplication tensor of rank n*m*p.
    """
    n, m, p = DIMENSIONS
    rank = n * m * p
    
    # Initialize matrices with zeros
    # U: (n*m, rank)
    # V: (m*p, rank)
    # W: (p*n, rank)
    U = np.zeros((n * m, rank), dtype=int)
    V = np.zeros((m * p, rank), dtype=int)
    W = np.zeros((p * n, rank), dtype=int)
    
    # Fill the decomposition
    # The tensor T[i, j, k] corresponds to the product A[i, j] * B[j, k]
    # In the factorization T = U @ V^T @ W^T (conceptually for each slice),
    # the column c corresponds to a specific (i, j, k) triple.
    c = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # U row index is i*m + j
                U[i * m + j, c] = 1
                # V row index is j*p + k
                V[j * p + k, c] = 1
                # W row index is k*n + i
                W[k * n + i, c] = 1
                c += 1
                
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
