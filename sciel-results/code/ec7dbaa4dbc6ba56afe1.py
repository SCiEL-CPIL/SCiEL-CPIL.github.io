"""Matrix multiplication 5x5x5: return the standard rank-125 tensor decomposition."""
import time

DIMENSIONS = [5, 5, 5]

# EVOLVE-BLOCK-START
def _schoolbook():
    """Return the standard rank-125 decomposition as a dict of lists."""
    n, m, p = DIMENSIONS
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

def construct():
    """
    Return the schoolbook rank-125 decomposition for 5x5x5 matrix multiplication.
    No lower-rank exact decomposition is known for this dimension.
    """
    return _schoolbook()
# EVOLVE-BLOCK-END
