"""Tensor decomposition for 2x2 by 2x4 matrix multiplication."""

# EVOLVE-BLOCK-START
def construct():
    # Build a valid decomposition for 2x2 (A) times 2x4 (B) -> 2x4 (C).
    # Indexing:
    #   A has n=2 rows, m=2 columns  -> U has n*m = 4 rows
    #   B has m=2 rows, p=4 columns  -> V has m*p = 8 rows
    #   C has p=4 rows, n=2 columns  -> W has p*n = 8 rows
    # Trace convention: W[k*n + i, r]
    #
    # We use a structured construction that groups by the B column k and
    # A column j, allowing the two A-rows to be combined within each (j,k)
    # pair without increasing the rank beyond the schoolbook 16.
    n, m, p = 2, 2, 4
    rank = n * m * p  # 16
    U = [[0] * rank for _ in range(n * m)]  # 4 x 16
    V = [[0] * rank for _ in range(m * p)]  # 8 x 16
    W = [[0] * rank for _ in range(p * n)]  # 8 x 16

    col = 0
    for j in range(m):
        for k in range(p):
            for i in range(n):
                # U index: i * m + j
                U[i * m + j][col] = 1
                # V index: j * p + k
                V[j * p + k][col] = 1
                # W index: k * n + i
                W[k * n + i][col] = 1
                col += 1

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
