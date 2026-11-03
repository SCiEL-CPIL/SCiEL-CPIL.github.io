"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 4, 8]

# EVOLVE-BLOCK-START
import numpy as np

def construct():
    n, m, p = DIMENSIONS  # 3, 4, 8
    
    # The target tensor is T[i, k, j] = 1 if i == j else 0.
    # This is the Trace tensor structure.
    # We can decompose this with rank n (which is 3) using the following:
    # For r in 0..n-1:
    #   U[i, k, r] = 1 if i == r else 0
    #   V[k, j, r] = 1 if j == r else 0
    #   W[j, i, r] = 1 if j == r and i == r else 0
    
    # Rank is n = 3
    rank = n
    
    # U shape: (n * m, rank)
    U = [[0] * rank for _ in range(n * m)]
    # V shape: (m * p, rank)
    V = [[0] * rank for _ in range(m * p)]
    # W shape: (p * n, rank)
    W = [[0] * rank for _ in range(p * n)]
    
    for r in range(rank):
        # Fill U
        # U index: i * m + k
        # U[i, k, r] = 1 if i == r
        for k in range(m):
            u_idx = r * m + k
            U[u_idx][r] = 1
            
        # Fill V
        # V index: k * p + j
        # V[k, j, r] = 1 if j == r
        for k in range(m):
            v_idx = k * p + r
            V[v_idx][r] = 1
            
        # Fill W
        # W index: j * n + i
        # W[j, i, r] = 1 if j == r and i == r
        # Only one entry is non-zero: j=r, i=r
        w_idx = r * n + r
        W[w_idx][r] = 1

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
