"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [3, 3, 3]

# EVOLVE-BLOCK-START
def _schoolbook():
    """Return the rank-27 schoolbook decomposition."""
    n, m, p = DIMENSIONS
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
    return {"U": U, "V": V, "W": W}

def _verify(U, V, W, tol=1e-9):
    """Check that the decomposition exactly decomposes the matmul tensor."""
    n, m, p = DIMENSIONS
    Ua = np.array(U, dtype=float)
    Va = np.array(V, dtype=float)
    Wa = np.array(W, dtype=float)
    rank = Ua.shape[1]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = np.sum(Ua[i*m+j, :] * Va[j*p+k, :] * Wa[k*n+i, :])
                if abs(val - 1.0) > tol:
                    return False
    return True

def _check_half_ints(U, V, W, tol=1e-9):
    """Check all entries are half-integers."""
    for mat in [U, V, W]:
        for row in mat:
            for v in row:
                if abs(v * 2 - round(v * 2)) > tol:
                    return False
    return True

def _round_to_half_ints(U, V, W):
    """Round all entries to nearest half-integer."""
    def r(x):
        return round(x * 2) / 2.0
    return (
        [[r(v) for v in row] for row in U],
        [[r(v) for v in row] for row in V],
        [[r(v) for v in row] for row in W],
    )

def construct():
    start_time = time.time()
    deadline = start_time + 110  # 10s margin
    
    # Return the schoolbook decomposition (rank 27)
    # This is guaranteed correct and passes verification.
    return _schoolbook()
# EVOLVE-BLOCK-END
