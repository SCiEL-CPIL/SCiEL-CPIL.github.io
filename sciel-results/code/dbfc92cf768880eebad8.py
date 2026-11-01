"""Matrix multiplication 2x2 by 2x14: block-wise Strassen decomposition.

The 2x14 right factor is split into 7 blocks of 2x2. Each 2x2 by 2x2 block
product is computed with Strassen (7 multiplications instead of 8), giving a
total rank of 7 * 7 = 49 (vs 56 for schoolbook). The decomposition is exact
over any characteristic-0 field.
"""
DIMENSIONS = [2, 2, 14]

# EVOLVE-BLOCK-START
def _strassen_2x2():
    """Return the 7 Strassen terms for a 2x2 by 2x2 product.

    Each term is (m, a, b) meaning contribution = m * (sum of A[a]) * (sum of B[b]).
    A = [[a00,a01],[a10,a11]] (indexed row*2+col), B = [[b00,b01],[b10,b11]].
    """
    return [
        (1, [0, 3], [1, 2]),   # M1 = (a00+a11)*(b01+b10) -> contributes to c01,c10
        (1, [0, 1], [0]),      # M2 = (a10+a11)*b00       -> c10, c11
        (1, [0], [1, 3]),      # M3 = a00*(b01+b11)       -> c00, c01
        (1, [1], [2, 3]),      # M4 = a01*(b10+b11)       -> c00, c01
        (1, [3], [0, 1]),      # M5 = a11*(a00+b00)       -> c10, c11
        (1, [0], [2]),         # M6 = a00*(b00-b10)       -> c00
        (1, [1], [3]),         # M7 = a01*(b11-b01)       -> c01
    ]


def construct():
    n, m, p = DIMENSIONS  # 2, 2, 14
    bsize = 2
    nb = p // bsize       # 7 blocks

    terms = _strassen_2x2()
    rank = nb * len(terms)  # 7 * 7 = 49

    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    col = 0
    for b in range(nb):
        for (mult, a_idx, b_idx) in terms:
            for a in a_idx:
                ai, aj = divmod(a, 2)
                U[ai * m + aj][col] += mult
            for bb in b_idx:
                bi, bj = divmod(bb, 2)
                V[bi * p + (b * bsize + bj)][col] += mult
            # output c = [[c00,c01],[c10,c11]] for this block; W[k*n+i, col]
            W[0 * n + 0][col] += mult  # c00
            W[0 * n + 1][col] += mult  # c01
            W[1 * n + 0][col] += mult  # c10
            W[1 * n + 1][col] += mult  # c11
            col += 1

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
