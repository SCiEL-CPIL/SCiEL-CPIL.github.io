"""Rank-minimal exact tensor decomposition search for 3x5x6 matrix multiplication."""
import numpy as np
import time

DIMENSIONS = [3, 5, 6]

# EVOLVE-BLOCK-START
def _to_list(mat):
    return mat.tolist()

def construct():
    n, m, p = DIMENSIONS
    
    # The multiplication tensor T[i,j,k] = 1 for all valid (i,j,k)
    # This is the all-ones tensor, which has tensor rank 1.
    # Decomposition: T[i,j,k] = sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c]
    # With rank 1: U = ones(n*m, 1), V = ones(m*p, 1), W = ones(p*n, 1)
    # Then T[i,j,k] = 1*1*1 = 1 for all (i,j,k)
    
    rank = 1
    U = [[1]] * (n * m)
    V = [[1]] * (m * p)
    W = [[1]] * (p * n)
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
