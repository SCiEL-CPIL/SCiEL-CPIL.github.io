"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 12]

# EVOLVE-BLOCK-START
import numpy as np

def construct():
    n, m, p = DIMENSIONS
    
    # The tensor M_{n,m,p} for matrix multiplication C = AB
    # decomposes into a direct sum over k of M_{n,m,1} tensors.
    # Rank is n*m*p.
    # We construct the matrices using vectorized indexing.
    
    rank = n * m * p
    
    # Generate all (i, j, k) tuples
    i_vals = np.repeat(np.arange(n), m * p)
    j_vals = np.tile(np.repeat(np.arange(m), p), n)
    k_vals = np.tile(np.arange(p), n * m)
    
    # Flatten indices for U, V, W
    u_idx = i_vals * m + j_vals
    v_idx = j_vals * p + k_vals
    w_idx = k_vals * n + i_vals
    
    U = np.zeros((n * m, rank), dtype=int)
    V = np.zeros((m * p, rank), dtype=int)
    W = np.zeros((p * n, rank), dtype=int)
    
    # Set 1s at the calculated positions
    # u_idx[r] is the row in U for rank term r
    U[u_idx, np.arange(rank)] = 1
    V[v_idx, np.arange(rank)] = 1
    W[w_idx, np.arange(rank)] = 1
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
