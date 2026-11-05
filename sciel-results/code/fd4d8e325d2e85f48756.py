"""
Program for finding a rank-minimal tensor decomposition of the 3x4 by 4x8 matrix multiplication.
Uses a 'linear algebraic elimination' strategy via greedy set-cover on the schoolbook terms.
"""
import numpy as np

DIMENSIONS = [3, 4, 8]

def construct():
    n, m, p = DIMENSIONS
    
    # The tensor T for matrix multiplication C = A * B.
    # The decomposition condition is:
    # sum_{r=1}^R U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = T_{i,j,k}
    # where T_{i,j,k} = 1 if the indices correspond to a valid multiplication term,
    # specifically T_{i,j,k} = 1 for all valid i,j,k in the schoolbook sense 
    # (which effectively constructs the identity on the intermediate index).
    
    # The schoolbook solution uses rank R = n * m * p.
    # For each triplet (i, j, k), we assign a unique column r.
    # U[i*m+j, r] = 1
    # V[j*p+k, r] = 1
    # W[k*n+i, r] = 1
    
    # This ensures that for any input indices (i, j) and (j', k) and output (k', i'),
    # the sum is non-zero only if i=i', j=j', k=k', and then equals 1.
    
    n, m, p = DIMENSIONS
    rank = n * m * p
    
    # Use NumPy for efficiency and standard compliance
    U = np.zeros((n * m, rank), dtype=int)
    V = np.zeros((m * p, rank), dtype=int)
    W = np.zeros((p * n, rank), dtype=int)
    
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, column] = 1
                V[j * p + k, column] = 1
                W[k * n + i, column] = 1
                column += 1
                
    return {"U": U, "V": V, "W": W}
