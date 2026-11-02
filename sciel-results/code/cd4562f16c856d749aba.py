"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank = 48
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    col = 0

    # Part 1: j = 0, standard 2x1x6, one term per k
    for k in range(p):
        for i in range(n):
            U[i * m][col] = 1.0
        V[k][col] = 1.0
        for i in range(n):
            W[k * n + i][col] = 1.0
        col += 1

    # Part 2: j in {1,2} and {3,4}, k in pairs, Strassen 2x2x2
    strassen_terms = [
        (
            [(0, 0, 1.0), (1, 1, 1.0)],
            [(0, 0, 1.0), (1, 1, 1.0)],
            [(0, 0, 1.0), (1, 1, 1.0)]
        ),
        (
            [(1, 0, 1.0), (1, 1, 1.0)],
            [(0, 0, 1.0)],
            [(1, 0, 1.0), (1, 1, -1.0)]
        ),
        (
            [(0, 0, 1.0)],
            [(0, 1, 1.0), (1, 1, -1.0)],
            [(0, 1, 1.0), (1, 1, 1.0)]
        ),
        (
            [(1, 1, 1.0)],
            [(1, 0, 1.0), (0, 0, -1.0)],
            [(0, 0, 1.0), (1, 0, 1.0)]
        ),
        (
            [(0, 0, 1.0), (0, 1, 1.0)],
            [(1, 1, 1.0)],
            [(0, 0, -1.0), (0, 1, 1.0)]
        ),
        (
            [(1, 0, 1.0), (0, 0, -1.0)],
            [(0, 0, 1.0), (0, 1, 1.0)],
            [(1, 1, 1.0)]
        ),
        (
            [(0, 1, 1.0), (1, 1, -1.0)],
            [(1, 0, 1.0), (1, 1, 1.0)],
            [(0, 0, 1.0)]
        ),
    ]

    for block_idx in range(2):
        j_base = 1 + block_idx * 2
        for s in range(3):
            k_base = s * 2
            for term in strassen_terms:
                u_list, v_list, w_list = term
                for (li, lj, val) in u_list:
                    gi = li
                    gj = j_base + lj
                    U[gi * m + gj][col] += val
                for (lj, lk, val) in v_list:
                    gj = j_base + lj
                    gk = k_base + lk
                    V[gj * p + gk][col] += val
                for (li, lk, val) in w_list:
                    gi = li
                    gk = k_base + lk
                    W[gk * n + gi][col] += val
                col += 1

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
