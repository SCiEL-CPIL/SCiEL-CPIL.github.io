"""Matrix multiplication 2,3,9 - rank-minimal exact tensor decomposition."""
import numpy as np

DIMENSIONS = [2, 3, 9]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    
    # The tensor is a direct sum of two T_{1,3,9} tensors.
    # Rank of T_{1,3,9} is m*p = 27.
    # Total rank is 2 * 27 = 54.
    rank = 2 * m * p
    
    U = np.zeros((n * m, rank), dtype=int)
    V = np.zeros((m * p, rank), dtype=int)
    W = np.zeros((p * n, rank), dtype=int)
    
    # Construct for each row i of the 2x3 matrix
    for i in range(n):
        # Offset for this block in the rank dimension
        r_offset = i * (m * p)
        
        for j in range(m):
            for k in range(p):
                r_idx = r_offset + j * p + k
                
                # U index: i*m + j
                U[i * m + j, r_idx] = 1
                # V index: j*p + k
                V[j * p + k, r_idx] = 1
                # W index: k*n + i
                W[k * n + i, r_idx] = 1
                
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
