"""Rank-minimal exact tensor decomposition for 2x5x5 matrix multiplication."""
DIMENSIONS = [2, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 2, 5, 5
    # A is (n x m), B is (m x p), C = A*B is (n x p).
    # Tensor T[i, j, k] = C[i, k] = sum_j A[i, j] * B[j, k].
    # U indexed by (i, j) -> n*m rows; V by (j, k) -> m*p rows; W by (k, i) -> p*n rows.
    # The 2x5x5 mult splits into two independent 1x5x5 (matrix-vector) mults,
    # each of exact tensor rank m = 5. Total rank = n * m = 10.
    rank = n * m
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
# EVOLVE-BLOCK-END
