"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 9]

import time
import random
import numpy as np
from itertools import product

# EVOLVE-BLOCK-START
def _schoolbook():
    """Return the standard rank-144 schoolbook decomposition for 4x4 * 4x9."""
    n, m, p = DIMENSIONS  # 4, 4, 9
    rank = n * m * p  # 144

    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # A[i,j] -> i*m+j ; B[j,k] -> j*p+k ; C[i,k] -> k*n+i (official trace convention)
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1

    return {"U": U, "V": V, "W": W}

def _verify_decomposition(U, V, W, n, m, p):
    """Verify that the decomposition exactly satisfies the tensor identity."""
    R = len(U[0]) if U else 0
    U2 = np.array([[int(round(u * 2)) for u in row] for row in U])
    V2 = np.array([[int(round(v * 2)) for v in row] for row in V])
    W2 = np.array([[int(round(w * 2)) for w in row] for row in W])
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0
                for r in range(R):
                    s += U2[i * m + j, r] * V2[j * p + k, r] * W2[k * n + i, r]
                if s != 8:
                    return False
    return True

def construct():
    """
    Returns a tensor decomposition for 4x4 * 4x9 matrix multiplication.
    
    Returns the guaranteed-valid schoolbook decomposition (rank 144).
    """
    n, m, p = DIMENSIONS
    return _schoolbook()
# EVOLVE-BLOCK-END
