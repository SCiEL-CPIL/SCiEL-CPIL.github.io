"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 3, 5]

import time
import numpy as np

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    
    # Structured decomposition with rank = m * p
    # For each (j, k) pair, create one rank-1 factor:
    # U[i*m + j, r(j,k)] = 1 for all i
    # V[j*p + k, r(j,k)] = 1
    # W[k*n + i, r(j,k)] = 1 for all i
    # This gives T[i*m+j, j*p+k, k*n+i] = 1 exactly.
    
    rank = m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    for j in range(m):
        for k in range(p):
            for i in range(n):
                U[i * m + j][col] = 1
                W[k * n + i][col] = 1
            V[j * p + k][col] = 1
            col += 1
    
    # Verify correctness
    U_np = np.array(U, dtype=int)
    V_np = np.array(V, dtype=int)
    W_np = np.array(W, dtype=int)
    
    T_computed = np.einsum('ar,br,cr->abc', U_np, V_np, W_np, optimize=True)
    
    T_target = np.zeros((n * m, m * p, p * n), dtype=int)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T_target[a, b, c] = 1
    
    if not np.array_equal(T_computed, T_target):
        # Fallback to schoolbook if verification fails
        rank_schoolbook = n * m * p
        U_fb = [[0] * rank_schoolbook for _ in range(n * m)]
        V_fb = [[0] * rank_schoolbook for _ in range(m * p)]
        W_fb = [[0] * rank_schoolbook for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_fb[i * m + j][col] = 1
                    V_fb[j * p + k][col] = 1
                    W_fb[k * n + i][col] = 1
                    col += 1
        return {"U": U_fb, "V": V_fb, "W": W_fb}
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
