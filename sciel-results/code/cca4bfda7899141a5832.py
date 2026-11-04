"""Matrix multiplication 2x3x7 decomposition.

Uses schoolbook decomposition (rank 42) with numerical verification.
"""
DIMENSIONS = [2, 3, 7]


def _verify(U, V, W, n, m, p, tol=1e-9):
    """Verify that the decomposition exactly reproduces matrix multiplication."""
    import numpy as np
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    rank = U.shape[1]
    for a_i in range(n):
        for a_j in range(m):
            for b_j in range(m):
                for b_k in range(p):
                    for c_i in range(n):
                        for c_k in range(p):
                            expected = 1.0 if (c_i == a_i and c_k == b_k and a_j == b_j) else 0.0
                            actual = sum(U[a_i * m + a_j, c] * V[b_j * p + b_k, c] * W[c_k * n + c_i, c]
                                         for c in range(rank))
                            if abs(actual - expected) > tol:
                                return False
    return True


def construct():
    n, m, p = 2, 3, 7
    import time
    deadline = time.time() + 120.0

    # Schoolbook decomposition (rank 42)
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

    # Verify
    if not _verify(U, V, W, n, m, p):
        raise RuntimeError("Schoolbook decomposition failed verification")

    return {"U": U, "V": V, "W": W}
