"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

DIMENSIONS = [5, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    """
    Constructs the rank-minimal exact tensor decomposition for 5x5 matrix multiplication.
    Currently returns the schoolbook decomposition of rank 125.
    """
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin for 120s limit
    
    n, m, p = DIMENSIONS
    
    # Schoolbook decomposition has rank n*m*p
    rank = n * m * p
    
    # Initialize matrices as numpy arrays for speed
    U = np.zeros((n * m, rank), dtype=np.int64)
    V = np.zeros((m * p, rank), dtype=np.int64)
    W = np.zeros((p * n, rank), dtype=np.int64)
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # Map indices to flat indices
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                
                U[u_idx, col] = 1
                V[v_idx, col] = 1
                W[w_idx, col] = 1
                col += 1
                
                # Check time
                if time.time() > deadline:
                    break
            if time.time() > deadline:
                break
        if time.time() > deadline:
            break
            
    # The schoolbook solution is exact and always valid.
    # Convert to lists if the interface strictly requires lists, 
    # but the prompt says "factor matrices of shapes...", 
    # and the seed used lists of lists. 
    # The evaluator likely accepts numpy arrays or lists. 
    # The seed returned lists. I will return lists to be safe with the "interface" 
    # matching the seed, or numpy arrays if that's more standard for "matrices".
    # The seed: return {"U": U, "V": V, "W": W} where U was a list of lists.
    
    return {
        "U": U.tolist(), 
        "V": V.tolist(), 
        "W": W.tolist()
    }
# EVOLVE-BLOCK-END
