"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 3, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = 3, 3, 6
    # Split B into even columns (0,2,4) and odd columns (1,3,5),
    # each forming a 3x3 matrix multiplication with A.
    # Schoolbook: 9 products per 3x3x3 block, total rank = 18.
    rank = 18
    U = [[0] * rank for _ in range(n * m)]   # shape (9, 18)
    V = [[0] * rank for _ in range(m * p)]   # shape (18, 18)
    W = [[0] * rank for _ in range(p * n)]   # shape (18, 18)

    col = 0
    # Even columns of B: k = 0, 2, 4
    for i in range(n):
        for j in range(m):
            for kk in range(3):
                k = 2 * kk
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1

    # Odd columns of B: k = 1, 3, 5
    for i in range(n):
        for j in range(m):
            for kk in range(3):
                k = 2 * kk + 1
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
