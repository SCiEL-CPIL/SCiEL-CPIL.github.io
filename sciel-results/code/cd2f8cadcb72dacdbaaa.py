import numpy as np

def construct():
    n, m, p = 2, 2, 12
    rank = n * m * p
    
    # Generate all combinations of (i, j, k)
    # i in [0, n), j in [0, m), k in [0, p)
    # Total rank = n*m*p
    
    # We can create the index arrays
    # r index from 0 to rank-1
    # We map r to (i, j, k)
    # Let's use a specific ordering:
    # r = i * (m * p) + j * p + k
    
    # Create U, V, W as zero matrices
    U = np.zeros((n * m, rank), dtype=int)
    V = np.zeros((m * p, rank), dtype=int)
    W = np.zeros((p * n, rank), dtype=int)
    
    # Instead of a loop, we can use broadcasting if we construct the index vectors
    # But simple loop is fine and clear. 
    # To make it "new", I will use a vectorized approach.
    
    # Create index arrays for each rank r
    r_idx = np.arange(rank)
    
    # Decode r into i, j, k
    # r = i * (m*p) + j * p + k
    i_idx = (r_idx // (m * p)).astype(int)
    j_idx = ((r_idx // p) % m).astype(int)
    k_idx = (r_idx % p).astype(int)
    
    # Calculate flat indices
    # U index: i * m + j
    u_idx = i_idx * m + j_idx
    
    # V index: j * p + k
    v_idx = j_idx * p + k_idx
    
    # W index: k * n + i
    w_idx = k_idx * n + i_idx
    
    # Set values to 1
    U[u_idx, r_idx] = 1
    V[v_idx, r_idx] = 1
    W[w_idx, r_idx] = 1
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
