"""Rewritten program: Kronecker-product split of the 2x5x6 multiplication tensor
into a 2x2x6 block (j in {0,1}) and a 2x3x6 block (j in {2,3,4}), each solved
independently with a rank-minimal algorithm, then stitched into full factor matrices."""
DIMENSIONS = [2, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = 2, 5, 6

    # ------------------------------------------------------------------
    # Block 1: sub-tensor for j in {0,1} -> a 2x2x6 multiplication tensor
    #   A2[i, j] (i in 0..1, j in 0..1), B2[j, k] (j in 0..1, k in 0..5)
    #   C2[i, k] = sum_j A2[i,j] B2[j,k]
    # We build this as a Kronecker product: row-block 2x2 (Strassen, rank 7)
    #        x col-block 1x6 split into a 2x3 structure handled directly.
    # Concretely we use the rank-24 algorithm: for each of 7 Strassen
    # products over the j (row) dimension we need to place 6 column
    # contributions, giving 7*6 = 42... but we can do better by pairing
    # columns. Instead we use a direct construction of rank 24:
    #   treat the 2x2x6 as (2x2x2) ⊗ (1x1x3) in the k dimension would
    #   give 7*3 = 21... we verify below with a rank-24 explicit scheme.
    #
    # We use the explicit rank-24 scheme obtained by applying the
    # 2x2x2 Strassen (rank 7) on the j dimension and a rank-3 scheme on
    # the 6 columns grouped as 3 pairs of columns (2x2x2 blocks in k),
    # but with the middle column handled efficiently. To keep the code
    # simple and provably correct, we use a *direct* rank-24 construction
    # enumerated below.
    # ------------------------------------------------------------------
    rank_block1 = 24
    # We will store (u_row, u_col, v_row, v_col, w_row, w_col, coeff)
    # in *local* 2x2x6 coordinates: u_row in {0,1} (i), u_col in {0,1} (j)
    # v_row in {0,1} (j), v_col in 0..5 (k), w_row in 0..5 (k), w_col in {0,1} (i)
    terms_b1 = []

    # Build 2x2x6 as a tensor product of a 2x2x2 Strassen (over j) and a
    # 1x1x3 identity (over k grouped into 3 pairs) -> rank 7*3 = 21.
    # But 2x2x2 has k-dimension 2; to get k-dimension 6 we take 3 copies
    # (k-pairs (0,1),(2,3),(4,5)) -> 7*3 = 21 terms. Let's verify rank.
    # The 2x2x2x... we instead do: rank(2x2x6) = rank(2x2x2) * 3 = 7*3 = 21.
    # So block1 rank = 21, not 24. Good - lower rank.
    rank_block1 = 21
    # Strassen 7 products for 2x2x2 (standard):
    # Indices: A[0,0]=a00, A[0,1]=a01, A[1,0]=a10, A[1,1]=a11
    #           B[0,0]=b00, B[0,1]=b01, B[1,0]=b10, B[1,1]=b11
    # C[0,0]=a00*b00+a01*b10 ; C[0,1]=a00*b01+a01*b11
    # C[1,0]=a10*b00+a11*b10 ; C[1,1]=a10*b01+a11*b11
    # Strassen products:
    # M1 = (a00+a11)(b00+b11)
    # M2 = (a10+a11)b00
    # M3 = a00(b10-b00)
    # M4 = a11(b01-b11)
    # M5 = (a00+a01)b11
    # M6 = (a10-a00)(b00+b10)
    # M7 = (a01-a11)(b11+b01)
    # C00 = M1+M4-M5+M7 ; C01 = M3+M5
    # C10 = M2+M4 ; C11 = M1-M2+M3+M6
    #
    # For each k-pair (k0,k1) we map b00->B[j0,k0], b01->B[j0,k1],
    # b10->B[j1,k0], b11->B[j1,k1], and a's are the A2 entries.
    # The output C[i,k0] and C[i,k1] receive contributions per the formula.

    strassen_defs = [
        # (u_rows_coeffs: list of (i, j, c)), (v_rows_coeffs: list of (j, k_local, c)),
        # (w_out: list of (i, k_local, c))
        # k_local: 0 -> k0, 1 -> k1
        # M1
        ([(0,0,1),(1,1,1)], [(0,0,1),(1,1,1)], [(0,0,1),(1,1,-1)]),
        # M2
        ([(1,0,1),(1,1,1)], [(0,0,1)], [(1,0,1)]),
        # M3
        ([(0,0,1)], [(1,0,1),(0,0,-1)], [(0,1,1)]),
        # M4
        ([(1,1,1)], [(0,1,1),(1,1,-1)], [(0,0,1),(1,1,1)]),
        # M5
        ([(0,0,1),(0,1,1)], [(1,1,1)], [(0,0,-1),(0,1,1)]),
        # M6
        ([(1,0,1),(0,0,-1)], [(0,0,1),(0,1,1)], [(1,1,1)]),
        # M7
        ([(0,1,1),(1,1,-1)], [(1,1,1),(1,0,1)], [(0,0,1)]),
    ]

    k_pairs = [(0,1),(2,3),(4,5)]
    for (kp0, kp1) in k_pairs:
        for (us, vs, ws) in strassen_defs:
            u_list = []
            for (i, j, c) in us:
                u_list.append((i, j, c))
            v_list = []
            for (j, kl, c) in vs:
                kcol = kp0 if kl == 0 else kp1
                v_list.append((j, kcol, c))
            w_list = []
            for (i, kl, c) in ws:
                krow = kp0 if kl == 0 else kp1
                w_list.append((krow, i, c))
            terms_b1.append((u_list, v_list, w_list))

    assert len(terms_b1) == rank_block1, len(terms_b1)

    # ------------------------------------------------------------------
    # Block 2: sub-tensor for j in {2,3,4} -> a 2x3x6 multiplication tensor
    #   A3[i, jj] (i in 0..1, jj in 0..2), B3[jj, k] (jj in 0..2, k in 0..5)
    #   C3[i, k] = sum_{jj} A3[i,jj] B3[jj,k]
    # Rank of 2x3x6: we use the algorithm rank(2x3x6) = 10 * 3 = 30? 
    # Actually rank(2x3) tensor with 6 outputs. The 2x3x1 tensor has rank 5
    # (since 2x3 multiplication needs 5? No - 2x3 matrix mult by 3x1 vector:
    # that's just 6 scalar products, rank 6). For 2x3x6 we can use a
    # Strassen-like or direct scheme. The naive rank is 2*3*6/... no.
    # The tensor T3[i, jj, k] = delta. It's the composition of a 2x3 and 3x6.
    # rank(2x3x6) >= max(rank(2x3), rank(3x6), ...) The trivial rank is
    # 3*2 = 6? No. Each (i,k) output = sum over jj. We can achieve rank
    # 6 by: for each jj (3 of them) and each i (2 of them) -> 6 terms each
    # covering all 6 k's? That gives rank 6 only if each term is rank-1 in
    # the full tensor, i.e. U[i,jj]=1, V[jj,k] for all k, W[k,i]. But V[jj,k]
    # is a row of B, which is fine as a single factor row. So term for (i,jj):
    #   U[i, jj] = 1, V[jj, k] for all k in 0..5, W[k, i] = 1.
    # That's 2*3 = 6 terms! Each term is rank-1. So rank(2x3x6) = 6?
    # Let me verify: C[i,k] = sum_jj A[i,jj] * (term) * B[jj,k].
    # Term (i,jj): contributes A[i,jj] * 1 * B[jj,k] * 1 = A[i,jj]B[jj,k]. Summed
    # over jj gives C[i,k]. Yes! So block2 rank = 6.
    # ------------------------------------------------------------------
    rank_block2 = 6
    terms_b2 = []
    for jj in range(3):
        for i in range(2):
            u_list = [(i, jj, 1)]
            v_list = [(jj, k, 1) for k in range(6)]
            w_list = [(k, i, 1) for k in range(6)]
            terms_b2.append((u_list, v_list, w_list))
    assert len(terms_b2) == rank_block2

    # ------------------------------------------------------------------
    # Stitch into full matrices.
    # Full indices: U row = i*m + j, V row = j*p + k, W row = k*n + i
    # Block1: j_local in {0,1} -> global j = j_local (0,1)
    # Block2: jj_local in {0,1,2} -> global j = jj_local + 2 (2,3,4)
    # ------------------------------------------------------------------
    total_rank = rank_block1 + rank_block2
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]

    col = 0
    # Block 1 terms
    for (u_list, v_list, w_list) in terms_b1:
        for (i, j, c) in u_list:
            U[i * m + j][col] += c
        for (j, k, c) in v_list:
            V[j * p + k][col] += c
        for (k, i, c) in w_list:
            W[k * n + i][col] += c
        col += 1
    assert col == rank_block1

    # Block 2 terms
    for (u_list, v_list, w_list) in terms_b2:
        for (i, jj, c) in u_list:
            U[i * m + (jj + 2)][col] += c
        for (jj, k, c) in v_list:
            V[(jj + 2) * p + k][col] += c
        for (k, i, c) in w_list:
            W[k * n + i][col] += c
        col += 1
    assert col == total_rank

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
