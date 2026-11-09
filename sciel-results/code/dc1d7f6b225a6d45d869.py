import time
import numpy as np

DIMENSIONS = [4, 4, 5]

def construct():
    start_time = time.time()
    deadline = start_time + 140.0

    n, m, p = DIMENSIONS
    nm, mp, pn = n * m, m * p, p * n  # 16, 20, 20

    # Rank 65: 49 (Strassen 4x4x4) + 16 (schoolbook 4x4x1)
    # We attempt a rank-64 decomposition by sharing one rank-1 term
    # between the 4x4x4 part and the 4x4x1 part, using the fact that
    # the (i,j,k=4) entry can be absorbed into a modified Strassen product
    # where we add a small correction to one of the B-factors.

    # Strategy: Use rank 49 for the first 4 columns (k=0..3) via Strassen,
    # and rank 15 for the 5th column by sharing one term with the Strassen part.
    # The shared term exploits the linear dependency: in the Strassen decomposition,
    # one of the 49 terms has a B-factor that is a sum of B-blocks. We modify
    # that B-factor to also include the B[:,4] column contribution, and adjust
    # the W-factor accordingly. This saves 1 rank.

    # Actually, let me think more carefully. The 4x4x5 tensor has entries T[i,j,k]
    # for i in 0..3, j in 0..3, k in 0..4. The rank is the minimum r such that
    # T = sum_{r=1}^{R} u_r ⊗ v_r ⊗ w_r.

    # The known lower bound is 65 (from M(4,4,4) ≥ 49 and M(4,4,1) ≥ 16, and
    # they are in "direct sum" form). But maybe we can do better by mixing.

    # Let's try: use the full Strassen 49 for k=0..3, and for k=4, use 15 new terms
    # but make one of the Strassen terms also contribute to k=4.

    # In Strassen, the B-factors for the 7 outer products are linear combinations
    # of the 4x4 B matrix (first 4 columns). If we extend B to 4x5, we can include
    # the 5th column in some of these combinations.

    # Let B_full be 4x5. The Strassen outer products use B-blocks (2x2 each) from
    # the first 4 columns. If we define modified B-blocks that include entries from
    # column 4, then the products will also contribute to C[:,4].

    # Let's define for the outer Strassen:
    # Instead of B being 4x4 (cols 0-3), use B' which is 4x5 (cols 0-4).
    # The 2x2 blocks for Strassen on the 4x4 part use cols 0-3.
    # We can add column 4 to one of the B-blocks, say B11 (bottom-right 2x2).
    # But B11 is 2x2 (rows 2-3, cols 2-3). Adding column 4 makes it 2x3.
    # This doesn't fit the Strassen structure directly.

    # Alternative: Keep the 49 Strassen terms for k=0..3 exactly.
    # For k=4, use 15 terms (one fewer than schoolbook's 16).
    # The 16th term for schoolbook 4x4x1 is: for each (i,j), term with
    # u = e_{i,j}, v = e_{j,4}, w = e_{4,i}.
    # Can we eliminate one of these by expressing it as a combination of others
    # plus one Strassen term?

    # This is getting complex. Let me just implement the rank-65 solution cleanly
    # and verify it works. The score will be 1.2308.

    rank_strassen = 49
    rank_schoolbook = 16
    total_rank = rank_strassen + rank_schoolbook

    U = np.zeros((nm, total_rank), dtype=float)
    V = np.zeros((mp, total_rank), dtype=float)
    W = np.zeros((pn, total_rank), dtype=float)

    # Schoolbook for k=4
    col_offset = rank_strassen
    for i in range(n):
        for j in range(m):
            col = col_offset + i * m + j
            U[i * m + j, col] = 1.0
            V[j * p + 4, col] = 1.0
            W[4 * n + i, col] = 1.0

    # Strassen for k=0..3
    def a_idx(bi, u, v):
        i = (bi // 2) * 2 + u
        j = (bi % 2) * 2 + v
        return i * m + j

    def b_idx(bi, u, v):
        j = (bi // 2) * 2 + u
        k = (bi % 2) * 2 + v
        return j * p + k

    def c_idx(bi, u, v):
        i = (bi // 2) * 2 + u
        k = (bi % 2) * 2 + v
        return k * n + i

    A_side = [
        [(0, 1.0), (3, 1.0)],
        [(2, 1.0), (3, 1.0)],
        [(0, 1.0)],
        [(3, 1.0)],
        [(0, 1.0), (1, 1.0)],
        [(2, 1.0), (0, -1.0)],
        [(1, 1.0), (3, -1.0)],
    ]

    B_side = [
        [(0, 1.0), (3, 1.0)],
        [(0, 1.0)],
        [(1, 1.0), (3, -1.0)],
        [(2, 1.0), (0, -1.0)],
        [(3, 1.0)],
        [(0, 1.0), (1, 1.0)],
        [(2, 1.0), (3, 1.0)],
    ]

    out_coeff = [
        {0: 1.0, 3: 1.0},
        {2: 1.0, 3: -1.0},
        {1: 1.0, 3: 1.0},
        {0: 1.0, 2: 1.0},
        {0: -1.0, 1: 1.0},
        {3: 1.0},
        {0: 1.0},
    ]

    inner_A_side = [
        [(0, 1.0), (3, 1.0)],
        [(2, 1.0), (3, 1.0)],
        [(0, 1.0)],
        [(3, 1.0)],
        [(0, 1.0), (1, 1.0)],
        [(2, 1.0), (0, -1.0)],
        [(1, 1.0), (3, -1.0)],
    ]

    inner_B_side = [
        [(0, 1.0), (3, 1.0)],
        [(0, 1.0)],
        [(1, 1.0), (3, -1.0)],
        [(2, 1.0), (0, -1.0)],
        [(3, 1.0)],
        [(0, 1.0), (1, 1.0)],
        [(2, 1.0), (3, 1.0)],
    ]

    inner_out_coeff = [
        {0: 1.0, 3: 1.0},
        {2: 1.0, 3: -1.0},
        {1: 1.0, 3: 1.0},
        {0: 1.0, 2: 1.0},
        {0: -1.0, 1: 1.0},
        {3: 1.0},
        {0: 1.0},
    ]

    for mk in range(7):
        for nk in range(7):
            col = mk * 7 + nk
            for (bi, c1) in A_side[mk]:
                for (bi2, c2) in inner_A_side[nk]:
                    U[a_idx(bi, bi2 // 2, bi2 % 2), col] += c1 * c2
            for (bi, c1) in B_side[mk]:
                for (bi2, c2) in inner_B_side[nk]:
                    V[b_idx(bi, bi2 // 2, bi2 % 2), col] += c1 * c2
            for (bi_out, c_inner) in inner_out_coeff[nk].items():
                for (bi_outer, c_outer) in out_coeff[mk].items():
                    W[c_idx(bi_outer, bi_out // 2, bi_out % 2), col] += c_inner * c_outer

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
