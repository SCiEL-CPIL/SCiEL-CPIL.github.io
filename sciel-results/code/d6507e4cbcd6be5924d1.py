import numpy as np
import time

DIMENSIONS = [2, 3, 9]

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 140
    
    num_a = n * m  # 6
    num_b = m * p  # 27
    num_c = p * n  # 18
    
    # We found that the rank of the 2x3x1 block is 3.
    # Total rank is 9 * 3 = 27.
    
    rank = p * 3 # 27
    U = np.zeros((num_a, rank))
    V = np.zeros((num_b, rank))
    W = np.zeros((num_c, rank))
    
    for k in range(p):
        base = k * 3
        for r in range(3):
            col = base + r
            # V indices: j*p + k. j=r.
            V[r * p + k, col] = 1.0
            
            # W indices: k*n + i. i=0,1.
            W[k * n + 0, col] = 1.0
            W[k * n + 1, col] = 1.0
            
            # U indices: i*m + j. i=0,1. j=r.
            U[0 * m + r, col] = 1.0
            U[1 * m + r, col] = 1.0
            
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
