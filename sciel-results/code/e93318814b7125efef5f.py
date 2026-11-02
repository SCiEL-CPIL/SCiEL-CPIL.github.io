"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 3, 8]

# EVOLVE-BLOCK-START
import time

def _make_schoolbook(n, m, p):
    """Standard schoolbook tensor decomposition with rank n*m*p."""
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
    return {"U": U, "V": V, "W": W}

def _validate(U, V, W, n, m, p):
    """Check if the decomposition exactly reproduces the matrix multiplication tensor."""
    rank = len(U[0])
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                for c in range(rank):
                    val += U[u_idx][c] * V[v_idx][c] * W[w_idx][c]
                if abs(val - 1) > 1e-9:
                    return False
    return True

def construct():
    """
    Construct the rank-minimal (or best known) exact tensor decomposition
    for multiplying a 3x3 matrix by a 3x8 matrix.
    Returns a dict with keys 'U', 'V', 'W'.
    """
    n, m, p = DIMENSIONS
    return _make_schoolbook(n, m, p)
# EVOLVE-BLOCK-END
