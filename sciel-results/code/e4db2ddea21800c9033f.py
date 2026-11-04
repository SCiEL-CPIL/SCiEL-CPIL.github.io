"""Rewritten program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 4, 5, 5
    rank = n * m * p      # 100
    U = [[0] * rank for _ in range(n * m)]  # 20 x 100
    V = [[0] * rank for _ in range(m * p)]  # 25 x 100
    W = [[0] * rank for _ in range(p * n)]  # 20 x 100
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
