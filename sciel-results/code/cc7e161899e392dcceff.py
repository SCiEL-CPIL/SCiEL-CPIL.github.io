"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    """Return a valid exact tensor decomposition for M(3,4,5)."""
    n, m, p = DIMENSIONS
    r = n * m * p
    U = [[0] * r for _ in range(n * m)]
    V = [[0] * r for _ in range(m * p)]
    W = [[0] * r for _ in range(p * n)]
    idx = 0
    for a in range(n):
        for b in range(m):
            for c in range(p):
                U[a * m + b][idx] = 1
                V[b * p + c][idx] = 1
                W[c * n + a][idx] = 1
                idx += 1
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
