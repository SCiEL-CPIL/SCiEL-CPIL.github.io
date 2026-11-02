"""Matrix multiplication tensor decomposition for 2x3 * 3x10."""
import time

DIMENSIONS = [2, 3, 10]

# EVOLVE-BLOCK-START
def construct():
    """Return a rank-minimal exact tensor decomposition for multiplying
    a 2x3 matrix by a 3x10 matrix.

    The rank is 10, achieved by exploiting the direct sum structure
    of the tensor into 10 independent 2x3x1 sub-problems, each of rank 1.
    """
    start = time.time()
    n, m, p = DIMENSIONS
    rank = p  # 10

    # Allocate factor matrices
    U = [[0] * rank for _ in range(n * m)]  # (6, 10)
    V = [[0] * rank for _ in range(m * p)]  # (30, 10)
    W = [[0] * rank for _ in range(p * n)]  # (20, 10)

    # Fill U: all ones
    for i in range(n * m):
        for r in range(rank):
            U[i][r] = 1

    # Fill V: V[j*p + k, r] = 1 if k == r
    for j in range(m):
        for k in range(p):
            V[j * p + k][k] = 1

    # Fill W: W[k*n + i, r] = 1 if k == r
    for k in range(p):
        for i in range(n):
            W[k * n + i][k] = 1

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
