"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
import random
from itertools import product

DIMENSIONS = [3, 4, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    nm, mp, pn = n * m, m * p, p * n
    
    # Build the target tensor T
    # T[i*m+j, j*p+k, k*n+i] = 1
    T = np.zeros((nm, mp, pn))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m+j, j*p+k, k*n+i] = 1.0

    # Schoolbook decomposition as fallback
    def schoolbook():
        r = n * m * p
        U = np.zeros((nm, r))
        V = np.zeros((mp, r))
        W = np.zeros((pn, r))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j, col] = 1
                    V[j*p+k, col] = 1
                    W[k*n+i, col] = 1
                    col += 1
        return U, V, W

    U_s, V_s, W_s = schoolbook()
    best_rank = U_s.shape[1]
    best_U, best_V, best_W = U_s, V_s, W_s

    deadline = time.time() + 140  # safety margin

    def verify(U, V, W, tol=1e-10):
        S = np.einsum('ar,br,cr->abc', U, V, W)
        return np.allclose(S, T, atol=tol)

    # Try the algebraic rank-12 construction
    def try_algebraic_rank_12():
        r = 12
        U = np.zeros((nm, r))
        V = np.zeros((mp, r))
        W = np.zeros((pn, r))
        
        # r index corresponds to (i, j) where i in 0..2, j in 0..3
        for i in range(n):
            for j in range(m):
                r_idx = i * m + j
                U[i*m+j, r_idx] = 1
                # V: index j*p + k. For fixed i,j, V is 1 for all k?
                # Based on previous analysis, this likely fails, but let's try.
                for k in range(p):
                    V[j*p+k, r_idx] = 1
                # W: index k*n + i. For fixed i,j, W is 1 for all k?
                for k in range(p):
                    W[k*n+i, r_idx] = 1
        
        if verify(U, V, W):
            return U, V, W
        return None

    result = try_algebraic_rank_12()
    if result:
        best_U, best_V, best_W = result
        best_rank = 12
        return {"U": [[float(x) for x in row] for row in best_U],
                "V": [[float(x) for x in row] for row in best_V],
                "W": [[float(x) for x in row] for row in best_W]}

    # Fallback to schoolbook
    return {"U": [[float(x) for x in row] for row in best_U],
            "V": [[float(x) for x in row] for row in best_V],
            "W": [[float(x) for x in row] for row in best_W]}
# EVOLVE-BLOCK-END
