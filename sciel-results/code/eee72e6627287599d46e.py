import time
import numpy as np
from itertools import product

DEADLINE = time.time() + 140  # 140s deadline, 20s safety margin from 160s limit

def construct():
    n, m, p = 3, 4, 6
    # We construct a valid schoolbook decomposition (Rank 72).
    # This is a deterministic, exact, integer decomposition.
    # While not necessarily rank-minimal (Strassen can be better for larger N, 
    # for small rectangular matrices, block Strassen might not beat schoolbook 
    # due to overhead or lack of squareness, but 72 is a valid upper bound).
    # We use this to ensure a valid, non-crashing result.
    
    rank = n * m * p
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    curr_rank = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # A[i,j] * B[j,k] contributes to C[i,k]
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                
                U[u_idx, curr_rank] = 1.0
                V[v_idx, curr_rank] = 1.0
                W[w_idx, curr_rank] = 1.0
                curr_rank += 1
                
    # The rank used is exactly n*m*p
    # Return as lists of lists
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
