import time
import numpy as np

DIMENSIONS = [2, 3, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    nm = n * m  # 6
    mp = m * p  # 15
    pn = p * n  # 10

    # Rank 6 decomposition
    rank = 6
    
    U = np.zeros((nm, rank))
    V = np.zeros((mp, rank))
    W = np.zeros((pn, rank))
    
    # Terms 0 to 4: Handle A[0:2, 0:2] * B[0:2, :]
    # For each k in 0..4 (column of B, which is row of C)
    for r in range(5):
        # U: A[0:2, 0:2] -> indices 0,1,3,4
        U[0, r] = 1
        U[1, r] = 1
        U[3, r] = 1
        U[4, r] = 1
        
        # V: B[0:2, r] -> indices r, 5+r
        V[r, r] = 1
        V[5 + r, r] = 1
        
        # W: C[r, 0:2] -> indices 2*r, 2*r+1
        W[2 * r, r] = 1
        W[2 * r + 1, r] = 1
        
    # Term 5: Handle A[0:2, 2] * B[2, :]
    # U: A[0:2, 2] -> indices 2, 5
    U[2, 5] = 1
    U[5, 5] = 1
    
    # V: B[2, 0:5] -> indices 10, 11, 12, 13, 14
    V[10, 5] = 1
    V[11, 5] = 1
    V[12, 5] = 1
    V[13, 5] = 1
    V[14, 5] = 1
    
    # W: C[0:5, 0:2] -> all indices 0..9
    for k in range(5):
        for i in range(2):
            W[k * 2 + i, 5] = 1
            
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
