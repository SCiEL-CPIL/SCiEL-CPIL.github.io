"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 4, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Trivial rank-nmp decomposition (rank 72)
    r = n * m * p
    U = [[0.0] * r for _ in range(n * m)]
    V = [[0.0] * r for _ in range(m * p)]
    W = [[0.0] * r for _ in range(p * n)]
    idx = 0
    for k in range(p):
        for j in range(m):
            for i in range(n):
                U[i * m + j][idx] = 1.0
                V[j * p + k][idx] = 1.0
                W[k * n + i][idx] = 1.0
                idx += 1
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
