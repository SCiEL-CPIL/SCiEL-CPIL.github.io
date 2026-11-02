"""Find a low-rank exact tensor decomposition for 2x3 matrix times 3x6 matrix."""
DIMENSIONS = [2, 3, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = 2, 3, 6
    # Schoolbook decomposition of rank n*m*p = 36
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]  # 6 x 36
    V = [[0] * rank for _ in range(m * p)]  # 18 x 36
    W = [[0] * rank for _ in range(p * n)]  # 12 x 36
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
