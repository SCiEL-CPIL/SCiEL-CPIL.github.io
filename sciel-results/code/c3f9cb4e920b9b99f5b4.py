"""Matrix multiplication 3x4x6 with NumPy-based tensor decomposition."""
import numpy as np

DIMENSIONS = [3, 4, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    
    # Use schoolbook algorithm with rank n*m*p = 72
    # This is the trivial decomposition where each term A[i,j]*B[j,k] is a rank-1 component.
    rank = n * m * p
    
    # Initialize U, V, W as NumPy arrays for efficiency
    U = np.zeros((n * m, rank), dtype=np.int8)
    V = np.zeros((m * p, rank), dtype=np.int8)
    W = np.zeros((p * n, rank), dtype=np.int8)
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # U index: i*m + j
                # V index: j*p + k
                # W index: k*n + i
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
                
    # Convert to lists as per interface requirement (or keep as arrays if allowed? 
    # The prompt says "factor matrices of shapes... with integer/half-integer entries".
    # The seed returned lists of lists. I will return lists of lists to be safe.)
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
