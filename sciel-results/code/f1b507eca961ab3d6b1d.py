"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time

DIMENSIONS = [2, 2, 7]

# EVOLVE-BLOCK-START
def construct():
    # Deadline guard (construction is constant-time, but keep a safety margin).
    _t0 = time.time()
    _DEADLINE = 120.0  # evaluator wall-clock limit; stay well under it.

    # Standard matrix multiplication tensor decomposition.
    # Rank is n * m * p.
    # T[i,j,k] = 1 for all i,j,k.
    # Decomposition: sum_{i,j,k} e_{i,j}^U \otimes e_{j,k}^V \otimes e_{k,i}^W
    n, m, p = DIMENSIONS
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                r = i * m * p + j * p + k
                U[i * m + j][r] = 1
                V[j * p + k][r] = 1
                W[k * n + i][r] = 1
                
    # Safety: return immediately; construction is O(n*m*p).
    if time.time() - _t0 > _DEADLINE:
        return {"U": U, "V": V, "W": W}
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
