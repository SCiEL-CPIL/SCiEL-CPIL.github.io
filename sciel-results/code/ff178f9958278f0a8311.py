"""Improved rank-minimal decomposition for 2x5 @ 5x6 using Kronecker splits and Strassen."""
DIMENSIONS = [2, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Split m=5 into 1+2+2.
    # Part A: j=0 -> rank(2x1x6) = 12
    # Part B: j=1,2 -> rank(2x2x6) via 3 Strassen blocks on (2,6) -> 3*7 = 21
    # Part C: j=3,4 -> rank(2x2x6) via 3 Strassen blocks on (2,6) -> 3*7 = 21
    # Total rank = 12 + 21 + 21 = 54
    rank = 54

    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    col = 0

    # Part A: j=0 only
    for i in range(n):
        for k in range(p):
            U[i * m + 0][col] = 1
            V[0 * p + k][col] = 1
            W[k * n + i][col] = 1
            col += 1

    # Strassen terms for 2x2x2 block: (u_list, v_list, w_list)
    # u: list of (local_i, local_j, val)
    # v: list of (local_j, local_k, val)
    # w: list of (local_i, local_k, val)
    strassen_2x2x2 = [
        (
            [(0, 0, 1), (1, 1, 1)],
            [(0, 0, 1), (1, 1, 1)],
            [(0, 0, 1), (1, 1, 1)]
        ),
        (
            [(1, 0, 1), (1, 1, 1)],
            [(0, 0, 1)],
            [(1, 0, 1), (1, 1, -1)]
        ),
        (
            [(0, 0, 1)],
            [(0, 1, 1), (1, 1, -1)],
            [(0, 1, 1), (1, 1, 1)]
        ),
        (
            [(1, 1, 1)],
            [(1, 0, 1), (0, 0, -1)],
            [(0, 0, 1), (1, 0, 1)]
        ),
        (
            [(0, 0, 1), (0, 1, 1)],
            [(1, 1, 1)],
            [(0, 0, -1), (0, 1, 1)]
        ),
        (
            [(1, 0, 1), (0, 0, -1)],
            [(0, 0, 1), (0, 1, 1)],
            [(1, 1, 1)]
        ),
        (
            [(0, 1, 1), (1, 1, -1)],
            [(1, 0, 1), (1, 1, 1)],
            [(0, 0, 1)]
        ),
    ]

    # Part B and C: each is a 2x2x6 block, decomposed as 3 Strassen 2x2x2 blocks
    for block_j in [1, 3]:
        for k_block in range(3):
            k_base = k_block * 2
            for term in strassen_2x2x2:
                u_list, v_list, w_list = term

                for (li, lj, val) in u_list:
                    gi = li
                    gj = block_j + lj
                    U[gi * m + gj][col] += val

                for (lj, lk, val) in v_list:
                    gj = block_j + lj
                    gk = k_base + lk
                    V[gj * p + gk][col] += val

                for (li, lk, val) in w_list:
                    gi = li
                    gk = k_base + lk
                    W[gk * n + gi][col] += val

                col += 1

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
