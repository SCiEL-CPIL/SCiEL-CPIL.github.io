"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np

DIMENSIONS = [2, 3, 6]

# EVOLVE-BLOCK-START
def construct():
    """
    Constructs a valid tensor decomposition for 2x3x6 matrix multiplication.
    Uses the standard rank-36 decomposition (one rank-1 term per tensor entry).
    """
    n, m, p = DIMENSIONS  # 2, 3, 6
    
    rank = n * m * p
    # rank = 36
    
    # Initialize factor matrices
    # U: (n*m, rank) -> (6, 36)
    # V: (m*p, rank) -> (18, 36)
    # W: (p*n, rank) -> (12, 36)
    
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    # Index mapping helpers
    def u_idx(i, j):
        return i * m + j
    
    def v_idx(j, k):
        return j * p + k
    
    def w_idx(k, i):
        return k * n + i
    
    r = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # Each (i, j, k) corresponds to a rank-1 term
                U[u_idx(i, j), r] = 1.0
                V[v_idx(j, k), r] = 1.0
                W[w_idx(k, i), r] = 1.0
                r += 1
                
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
