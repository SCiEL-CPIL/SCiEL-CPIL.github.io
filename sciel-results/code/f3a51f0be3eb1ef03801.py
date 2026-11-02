"""Rank-72 schoolbook decomposition for 3x3x8 matrix multiplication."""
DIMENSIONS = [3, 3, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 3, 3, 8
    rank = n * m * p  # 72
    U = [[0] * rank for _ in range(n * m)]   # 9 x 72
    V = [[0] * rank for _ in range(m * p)]   # 24 x 72
    W = [[0] * rank for _ in range(p * n)]   # 24 x 72

    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1.0
                V[j * p + k][col] = 1.0
                W[k * n + i][col] = 1.0
                col += 1

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
