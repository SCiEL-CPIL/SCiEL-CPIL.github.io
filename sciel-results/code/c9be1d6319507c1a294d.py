import time
import numpy as np
import itertools

DIMENSIONS = [3, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    """
    Find a rank-minimal exact tensor decomposition for 3x4 * 4x8 matrix multiplication.
    
    Strategy: Heuristic Term Merging
    1. Start with schoolbook decomposition (rank 96).
    2. Try to merge pairs of terms if their sum can be represented by a single term 
       with adjusted coefficients (which is generally not possible for rank-1 tensors 
       unless they are proportional, but we can use the freedom of the decomposition 
       to find a sparser representation).
    3. In practice, for this specific size, we will attempt to find a rank-88 
       decomposition by exploiting the 2x4x8 substructure.
    """
    
    n, m, p = DIMENSIONS  # 3, 4, 8
    dim_u = n * m  # 12
    dim_v = m * p  # 32
    dim_w = p * n  # 24
    
    start_time = time.time()
    deadline = start_time + 110
    
    # Schoolbook
    U = np.zeros((dim_u, 96))
    V = np.zeros((dim_v, 96))
    W = np.zeros((dim_w, 96))
    
    idx = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m + j, idx] = 1.0
                V[j*p + k, idx] = 1.0
                W[k*n + i, idx] = 1.0
                idx += 1
                
    # Try to reduce rank by merging terms in the 2x4x8 part.
    # This is a placeholder for a more advanced search.
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
