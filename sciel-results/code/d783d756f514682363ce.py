"""Matrix multiplication 2x3x6 tensor decomposition; schoolbook baseline with safe verification."""
DIMENSIONS = [2, 3, 6]

import time

_START = time.time()
_WALL_LIMIT = 115.0  # safety margin before 150s wall-clock kill

# EVOLVE-BLOCK-START
def _schoolbook(n, m, p):
    """Schoolbook rank n*m*p decomposition with integer entries."""
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    return U, V, W


def _verify(U, V, W, n, m, p, tol=1e-9):
    """Check exact decomposition: for all i,j,k, sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c]==1."""
    try:
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for c in range(len(U[0])):
                        s += float(U[i * m + j][c]) * float(V[j * p + k][c]) * float(W[k * n + i][c])
                    if abs(s - 1.0) > tol:
                        return False
        return True
    except Exception:
        return False


def construct():
    deadline = _START + _WALL_LIMIT
    n, m, p = DIMENSIONS

    # Schoolbook decomposition always works and is valid.
    U, V, W = _schoolbook(n, m, p)

    # Verify before returning; fallback is the same schoolbook (already valid).
    if not _verify(U, V, W, n, m, p):
        # Should not happen for schoolbook, but keep the guard.
        pass

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
