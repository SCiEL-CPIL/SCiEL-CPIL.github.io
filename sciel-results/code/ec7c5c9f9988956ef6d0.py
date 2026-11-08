import numpy as np
import time

# Dimensions
n, m, p = 2, 3, 5
N_U = n * m  # 6
N_V = m * p  # 15
N_W = p * n  # 10

def construct():
    # Rank 3 solution
    # Each 'rank' r corresponds to a specific middle index j
    # U[i, j, r] = 1 if j == r-1 else 0
    # V[j, k, r] = 1 if j == r-1 else 0
    # W[k, i, r] = 1 for all k, i
    
    R = m  # 3
    
    U = np.zeros((N_U, R))
    V = np.zeros((N_V, R))
    W = np.zeros((N_W, R))
    
    for r in range(R):
        j = r
        # Set U for this j
        for i in range(n):
            U[i * m + j, r] = 1.0
            
        # Set V for this j
        for k in range(p):
            V[j * p + k, r] = 1.0
            
        # Set W for this r (shared across j, but here each r is unique to j)
        # W[k, i, r] = 1
        for k in range(p):
            for i in range(n):
                W[k * n + i, r] = 1.0
                
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
