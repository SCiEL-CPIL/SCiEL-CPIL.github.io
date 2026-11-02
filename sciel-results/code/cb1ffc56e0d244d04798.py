"""Construct a rank-12 tensor decomposition for 3x3 * 3x4 matrix multiplication."""
DIMENSIONS = [3, 3, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = 3, 3, 4
    rank = 12
    U = [[0] * rank for _ in range(n * m)]   # 9 rows
    V = [[0] * rank for _ in range(m * p)]   # 12 rows
    W = [[0] * rank for _ in range(p * n)]   # 12 rows

    # Terms 0-8: schoolbook for 3x3x3 (k=0,1,2)
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(3):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1

    # Terms 9-11: handle k=3 column, one term per output row i
    for i in range(n):
        r = 9 + i
        # U: all j entries are 1
        for j in range(m):
            U[i * m + j][r] = 1
        # V: all j entries at k=3 are 1
        for j in range(m):
            V[j * p + 3][r] = 1
        # W: only at (k=3, i)
        W[3 * n + i][r] = 1

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
