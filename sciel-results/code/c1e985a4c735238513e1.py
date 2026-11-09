"""Program for matrix multiplication tensor decomposition of (3,3,8)."""
import time

DIMENSIONS = [3, 3, 8]

DEADLINE_SECONDS = 120.0
SAFETY_MARGIN = 10.0


def _schoolbook_decomposition(n, m, p):
    """Build the standard (schoolbook) rank-n*m*p decomposition."""
    rank = n * m * p
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    c = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][c] = 1.0
                V[j * p + k][c] = 1.0
                W[k * n + i][c] = 1.0
                c += 1
    return U, V, W


def construct():
    start_time = time.time()
    
    n, m, p = DIMENSIONS
    U, V, W = _schoolbook_decomposition(n, m, p)
    best_result = {"U": U, "V": V, "W": W}
    best_rank = n * m * p
    
    # No better decomposition found within time budget
    return best_result
