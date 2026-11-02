"""Matrix multiplication tensor decomposition for 2x3 * 3x10."""
import time

DIMENSIONS = [2, 3, 10]

# EVOLVE-BLOCK-START
def construct():
    """Return a rank-minimal exact tensor decomposition for multiplying
    a 2x3 matrix by a 3x10 matrix.

    The schoolbook decomposition gives rank n*m*p.  For (2,3,10) this is 60.
    """
    start = time.time()
    n, m, p = DIMENSIONS
    rank = n * m * p  # 60

    # Allocate factor matrices as lists of lists
    U = [[0] * rank for _ in range(n * m)]  # (6, 60)
    V = [[0] * rank for _ in range(m * p)]  # (30, 60)
    W = [[0] * rank for _ in range(p * n)]  # (20, 60)

    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
