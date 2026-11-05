"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = 2, 3, 7
    rank = 8
    U = [[0.0] * rank for _ in range(n * m)]  # 6 x 8
    V = [[0.0] * rank for _ in range(m * p)]  # 21 x 8
    W = [[0.0] * rank for _ in range(p * n)]  # 14 x 8

    # Part 1: j in {0, 1} — 2x2x7 subproblem, 7 terms (one per k)
    for r in range(7):
        k = r
        # U: i in {0,1}, j in {0,1}
        for i in range(2):
            for j in range(2):
                U[i * m + j][r] = 1.0
        # V: j in {0,1}, k=r
        for j in range(2):
            V[j * p + k][r] = 1.0
        # W: k=r, i in {0,1}
        for i in range(2):
            W[k * n + i][r] = 1.0

    # Part 2: j = 2 — 2x1x7 subproblem, 1 term
    r = 7
    # U: i in {0,1}, j=2
    for i in range(2):
        U[i * m + 2][r] = 1.0
    # V: j=2, k in {0,...,6}
    for k in range(7):
        V[2 * p + k][r] = 1.0
    # W: k in {0,...,6}, i in {0,1}
    for k in range(7):
        for i in range(2):
            W[k * n + i][r] = 1.0

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
