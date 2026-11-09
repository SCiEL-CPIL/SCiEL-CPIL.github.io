"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 5]

# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin
    
    # Schoolbook rank
    rank_schoolbook = n * m * p
    nm, mp, pn = n * m, m * p, p * n
    
    # Initialize U, V, W as numpy arrays for speed, then convert to lists
    U = np.zeros((nm, rank_schoolbook), dtype=int)
    V = np.zeros((mp, rank_schoolbook), dtype=int)
    W = np.zeros((pn, rank_schoolbook), dtype=int)
    
    # Fill the schoolbook decomposition
    # U[i*m + j, r] = 1
    # V[j*p + k, r] = 1
    # W[k*n + i, r] = 1
    # where r corresponds to the triple (i, j, k)
    
    r = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                idx_u = i * m + j
                idx_v = j * p + k
                idx_w = k * n + i
                
                U[idx_u, r] = 1
                V[idx_v, r] = 1
                W[idx_w, r] = 1
                r += 1
                
    # Convert to lists as required by the interface
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}

# EVOLVE-BLOCK-END
