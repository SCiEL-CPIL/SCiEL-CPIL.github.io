"""Rewritten program for matrix multiplication tensor decomposition."""
import numpy as np
import time

DIMENSIONS = [2, 3, 10]

# EVOLVE-BLOCK-START
def construct():
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin
    
    n, m, p = DIMENSIONS
    
    # The tensor T is defined by T[i, j, k] = 1 for all i, j, k.
    # The schoolbook algorithm uses rank n*m*p.
    # Each rank-1 component corresponds to a unique (i, j, k) triple.
    # U, V, W are sparse identity-like structures mapped to the rank dimension.
    
    rank = n * m * p
    
    # Initialize as integer arrays to ensure exactness
    U = np.zeros((n * m, rank), dtype=np.int64)
    V = np.zeros((m * p, rank), dtype=np.int64)
    W = np.zeros((p * n, rank), dtype=np.int64)
    
    # Map each (i, j, k) to a unique rank index r
    # r = i * (m * p) + j * p + k
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                r = i * (m * p) + j * p + k
                # U maps (i, j) to row i*m+j
                U[i * m + j, r] = 1
                # V maps (j, k) to row j*p+k
                V[j * p + k, r] = 1
                # W maps (k, i) to row k*n+i (Trace convention W[k*n+i, r])
                W[k * n + i, r] = 1
                
                # Time check
                if time.time() > deadline:
                    break
            if time.time() > deadline:
                break
        if time.time() > deadline:
            break

    # Convert to list of lists as required by the interface
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
# EVOLVE-BLOCK-END
