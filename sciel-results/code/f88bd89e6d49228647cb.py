"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    U = [[0] * (m * p) for _ in range(n * m)]
    V = [[0] * (m * p) for _ in range(m * p)]
    W = [[0] * (m * p) for _ in range(p * n)]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                c = j * p + k
                U[i * m + j][c] = 1
                V[j * p + k][c] = 1
                W[k * n + i][c] = 1
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
