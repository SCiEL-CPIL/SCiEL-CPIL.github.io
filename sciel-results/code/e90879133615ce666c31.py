"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 4, 5]

# EVOLVE-BLOCK-START
import time
import numpy as np
from fractions import Fraction
import itertools

def construct():
    """
    Find a rank-minimal exact tensor decomposition for multiplying a 3 by 4 matrix by a 4 by 5 matrix.
    The output factor follows the official trace convention W[k*n+i,r].
    """
    n, m, p = DIMENSIONS
    
    # The decomposition condition:
    # For all i in [0,n), j in [0,m), k in [0,p):
    #   sum_{r=0}^{R-1} U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
    #
    # U: (n*m, R), V: (m*p, R), W: (p*n, R)
    # Entries must be integer or half-integer.
    
    # Schoolbook decomposition: rank = n*m*p = 60
    # Each term corresponds to one (i,j,k) triple
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j][col] = 1
                V[j*p+k][col] = 1
                W[k*n+i][col] = 1
                col += 1
    
    # Verify the decomposition
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0
                for r in range(rank):
                    s += U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r]
                assert s == 1, f"Verification failed at ({i},{j},{k}): got {s}"
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
