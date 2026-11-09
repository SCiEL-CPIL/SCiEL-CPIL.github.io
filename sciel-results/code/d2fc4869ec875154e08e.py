"""Matrix multiplication 2x4x8 via block-level Strassen decomposition."""
DIMENSIONS = [2, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 2, 4, 8
    # Block-level Strassen: split j into 2 groups of 2, k into 4 groups of 2
    # Each 2x2 * 2x2 sub-block uses Strassen rank 7
    # Total: 2 j-groups * 4 k-subgroups * 7 = 56
    R = 56

    # Precompute Strassen coefficient tables
    # P: for each M index, {(i_loc, j_loc): coeff}
    P_tbl = [
        {(0, 0): 1, (1, 1): 1},
        {(1, 0): 1, (1, 1): 1},
        {(0, 0): 1},
        {(1, 1): 1},
        {(0, 0): 1, (0, 1): 1},
        {(1, 0): 1, (0, 0): -1},
        {(0, 1): 1, (1, 1): -1},
    ]
    # Q: for each M index, {(j_loc, k_loc): coeff}
    Q_tbl = [
        {(0, 0): 1, (1, 1): 1},
        {(0, 0): 1},
        {(0, 1): 1, (1, 1): -1},
        {(1, 0): 1, (0, 0): -1},
        {(1, 1): 1},
        {(0, 0): 1, (0, 1): 1},
        {(1, 0): 1, (1, 1): 1},
    ]
    # R: for each M index, {(i_loc, k_loc): coeff}
    R_tbl = [
        {(0, 0): 1, (1, 1): 1},
        {(1, 0): 1, (1, 1): -1},
        {(0, 1): 1, (1, 1): 1},
        {(0, 0): 1, (1, 0): 1},
        {(0, 0): -1, (0, 1): 1},
        {(1, 1): 1},
        {(0, 0): 1},
    ]

    # Build factor matrices as lists of lists
    U = [[0] * R for _ in range(n * m)]   # (8, 56)
    V = [[0] * R for _ in range(m * p)]   # (32, 56)
    W = [[0] * R for _ in range(p * n)]   # (16, 56)

    col = 0
    for jg in range(2):
        jo = jg * 2
        for kg in range(4):
            ko = kg * 2
            for mi in range(7):
                for (il, jl), cp in P_tbl[mi].items():
                    U[il * m + jo + jl][col] += cp
                for (jl, kl), cq in Q_tbl[mi].items():
                    V[(jo + jl) * p + ko + kl][col] += cq
                for (il, kl), cr in R_tbl[mi].items():
                    W[(ko + kl) * n + il][col] += cr
                col += 1

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
