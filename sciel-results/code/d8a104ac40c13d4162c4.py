import numpy as np

DIMENSIONS = [2, 2, 16]

def construct():
    n, m, p = DIMENSIONS
    rank = n * m * p
    
    # Generate all indices
    i_vals = np.repeat(np.arange(n), m * p)
    j_vals = np.tile(np.repeat(np.arange(m), p), n)
    k_vals = np.tile(np.arange(p), n * m)
    
    # Calculate flat indices
    u_indices = i_vals * m + j_vals
    v_indices = j_vals * p + k_vals
    w_indices = k_vals * n + i_vals
    
    # Initialize
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    # Set values
    U[u_indices, np.arange(rank)] = 1.0
    V[v_indices, np.arange(rank)] = 1.0
    W[w_indices, np.arange(rank)] = 1.0
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
