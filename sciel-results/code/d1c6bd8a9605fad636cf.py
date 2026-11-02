"""Program for matrix multiplication 2x2x12 with rank-42 Strassen-based decomposition."""
DIMENSIONS = [2, 2, 12]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    num_blocks = p // m  # 6 blocks of 2x2
    rank_per_block = 7
    total_rank = num_blocks * rank_per_block

    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]

    # A indices (2x2): row*m + col
    ia, ib, ic, id_ = 0, 1, m, m + 1

    for k in range(num_blocks):
        base = k * rank_per_block
        c0, c1 = 2 * k, 2 * k + 1

        # B indices (2x12): row*p + col
        ie, if_, ig, ih = c0, c1, p + c0, p + c1

        # C indices (2x12): col*n + row  (trace convention W[k*n+i, r])
        ii, ik, ij, il = c0 * n, c0 * n + 1, c1 * n, c1 * n + 1

        # M1 = (a+d)(e+h)
        r = base + 0
        U[ia][r] = 1; U[id_][r] = 1
        V[ie][r] = 1; V[ih][r] = 1
        W[ii][r] = 1; W[il][r] = 1

        # M2 = (c+d)e
        r = base + 1
        U[ic][r] = 1; U[id_][r] = 1
        V[ie][r] = 1
        W[ik][r] = 1; W[il][r] = -1

        # M3 = a(f-h)
        r = base + 2
        U[ia][r] = 1
        V[if_][r] = 1; V[ih][r] = -1
        W[ij][r] = 1; W[il][r] = 1

        # M4 = d(g-e)
        r = base + 3
        U[id_][r] = 1
        V[ig][r] = 1; V[ie][r] = -1
        W[ii][r] = 1; W[ik][r] = 1

        # M5 = (a+b)h
        r = base + 4
        U[ia][r] = 1; U[ib][r] = 1
        V[ih][r] = 1
        W[ii][r] = -1; W[ij][r] = 1

        # M6 = (c-a)(e+f)
        r = base + 5
        U[ic][r] = 1; U[ia][r] = -1
        V[ie][r] = 1; V[if_][r] = 1
        W[il][r] = 1

        # M7 = (b-d)(g+h)
        r = base + 6
        U[ib][r] = 1; U[id_][r] = -1
        V[ig][r] = 1; V[ih][r] = 1
        W[ii][r] = 1

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
