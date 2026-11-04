"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
from fractions import Fraction
import time

DIMENSIONS = [2, 5, 6]

def _verify(U, V, W, n, m, p):
    """Verify that U, V, W form a valid decomposition."""
    rank = len(U[0]) if U else 0
    U_mat = np.array(U, dtype=object)
    V_mat = np.array(V, dtype=object)
    W_mat = np.array(W, dtype=object)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0
                for r in range(rank):
                    s += U_mat[i * m + j, r] * V_mat[j * p + k, r] * W_mat[k * n + i, r]
                expected = 1 if (i == j or j == k or i == k) else 0
                if s != expected:
                    return False
    return True

def _try_rank_bruteforce(R, n, m, p, deadline, time_budget):
    """Try to find a decomposition of rank R by random search."""
    # For small R, we can try random integer/half-integer entries
    # This is very unlikely to succeed for R < 60, but let's try
    pass

def construct():
    start_time = time.time()
    deadline = start_time + 120.0
    
    n, m, p = DIMENSIONS
    
    # Schoolbook decomposition of rank n*m*p
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    
    # Verify the schoolbook decomposition
    if not _verify(U, V, W, n, m, p):
        raise ValueError("Schoolbook decomposition failed verification")
    
    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
