"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
import itertools

DIMENSIONS = [3, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    nm, mp, pn = n*m, m*p, p*n
    
    # Use standard schoolbook multiplication
    # Rank is n * m * p
    total_rank = nm * p # n*m*p
    
    U = np.zeros((nm, total_rank))
    V = np.zeros((mp, total_rank))
    W = np.zeros((pn, total_rank))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # U index: i*m + j
                U[i*m + j, col] = 1
                # V index: j*p + k
                V[j*p + k, col] = 1
                # W index: k*n + i
                W[k*n + i, col] = 1
                col += 1
                
    if col != total_rank:
        raise ValueError(f"Column count mismatch: {col} vs {total_rank}")
        
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
