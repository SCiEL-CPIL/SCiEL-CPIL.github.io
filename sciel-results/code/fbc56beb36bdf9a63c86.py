"""Matrix multiplication 4x4x9 rank-minimal decomposition.

Uses a two-phase approach:
  Phase A – a fast, deterministic block construction that combines the known
    Strassen rank-7 2×2×2 core with linear padding to cover the 4×4×9
    problem in O(1) time (no iteration).
  Phase B – a bounded ALS + polish + rounding search that attempts to
    improve on the Phase-A rank if time permits.
"""
import numpy as np
import time

DIMENSIONS = [4, 4, 9]

# EVOLVE-BLOCK-START
def construct():
    t0 = time.time()
    deadline = t0 + 115          # safety margin before 120 s kill

    n, m, p = DIMENSIONS         # 4, 4, 9
    U_rows, V_rows, W_rows = n * m, m * p, p * n   # 16, 36, 36

    # ── Build target tensor ──────────────────────────────────────────────
    triples = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                triples.append((a, b, c))
    triple_set = set(triples)

    T_U = np.zeros((U_rows, V_rows, W_rows))
    for (a, b, c) in triples:
        T_U[a, b, c] = 1.0

    def exact_check(U, V, W, tol=1e-9):
        """Verify exact decomposition: all triples = 1, all non-triples = 0."""
        T_recon = np.einsum('ar,br,cr->abc', U, V, W)
        return bool(np.allclose(T_recon, T_U, atol=tol))

    # ── Schoolbook fallback (rank 144) ───────────────────────────────────
    rank_school = n * m * p
    U_school = np.zeros((U_rows, rank_school))
    V_school = np.zeros((V_rows, rank_school))
    W_school = np.zeros((W_rows, rank_school))
    for col, (a, b, c) in enumerate(triples):
        U_school[a, col] = 1
        V_school[b, col] = 1
        W_school[c, col] = 1

    # ── Phase A: Strassen-style block construction ───────────────────────
    #
    # Partition the 4×4 × 4×9 product into a 2×2 grid of sub-blocks:
    #   A = [[A11, A12],     B = [[B11, B12, B13],
    #        [A21, A22]]        [B21, B22, B23]]
    # where each Aij is 2×2, each Bij is 2×3.
    #
    # C = A·B, each Cij is 2×3.
    #
    # For each Cij = Aij · Bij  (2×2 times 2×3):
    #   Pad Bij to 2×2 by zeroing the third column: Bij_pad = [Bij | 0]
    #   Use Strassen rank-7 for the 2×2×2 core.
    #   The Strassen formula computes Aij · Bij_pad = Cij_padded.
    #   The first two columns of Cij_padded are exactly Cij.
    #
    # We need 4 sub-blocks, each of rank 7 → total rank 28.
    #
    # However, we must be careful: the Strassen decomposition is for
    # (X, Y, Z) where Z = X·Y, and the tensor entries are indexed as
    # T[x_idx, y_idx, z_idx] = 1 iff z_idx corresponds to x·y entry.
    #
    # In our global tensor, the indices are:
    #   a = i*m + j  (row of A: i = block-row, j = within-block row)
    #   b = j_block*p + k  (col of B: j_block = block-col, k = within-block col)
    #   c = k*n + i  (row of C: k = block-row-of-C, i = within-block col-of-C)
    #
    # Wait – the convention is W[k*n + i, r] where k is the C-row and i is
    # the C-col.  Let me re-derive.
    #
    # Standard: C = A·B, C is 4×9.
    #   C[rr, cc] = sum_j A[rr, j] * B[j, cc]
    #   a = rr*m + j = rr*4 + j
    #   b = j*p + cc = j*9 + cc
    #   c = cc*n + rr = cc*4 + rr
    #
    # Block decomposition: rr = 2*r + s (r,s ∈ {0,1}), cc = 2*c + d
    #   but cc goes 0..8, so c ∈ {0..3}, d ∈ {0,1} for cc<8, and cc=8 is c=4,d=0.
    #   Actually cc ∈ 0..8.  Let c = cc // 2, d = cc % 2.  Then c ∈ {0..4}.
    #   For c=4 (cc=8), d=0, and B[., 8] is the 9th column.
    #
    # This is getting complex.  Let's use a direct, verified construction.
    #
    # ── Direct verified construction: pad to 4×4×4, use Strassen recursion ─
    #
    # Pad B from 4×9 to 4×12 by zero-padding (columns 9-11).
    # Pad C from 4×9 to 4×12 similarly (but we only need first 9 cols).
    # Actually, let's pad to 4×4×8 (even dimensions) or 4×4×12.
    #
    # Better approach: use the known rank-19 for 4×4×4 (or rank-28 for
    # 4×4×4 via 2×2 Strassen blocks).  Then handle the 9 columns.
    #
    # Let's try: rank for 4×4×8.  Partition into 2×2 blocks of size
    # 2×2 and 2×4.  Each 2×2·2×4 = pad to 2×2·2×2, rank 7.  4 blocks → 28.
    # Then 4×4×9: the 9th column can be handled by adding 4 more terms
    # (one per row of A, column 8 of B) → rank 28 + 4 = 32.
    #
    # Actually, let's just try to construct and verify.

    def strassen_2x2x2():
        """Return (U, V, W) of shape (4,7) each for 2×2×2 matrix mult.

        Indices: a = i*2+j (i,j ∈ {0,1}), b = j*2+k, c = k*2+i
        where i = A-row, j = A-col/B-row, k = B-col/C-row.
        C-col = i.
        """
        U = np.zeros((4, 7))
        V = np.zeros((4, 7))
        W = np.zeros((4, 7))
        # Strassen factors (M1..M7), with A entries [a00,a01;a10,a11]
        # B entries [b00,b01;b10,b11], C entries [c00,c01;c10,c11]
        # M1 = (a00+a11)(b00+b11)
        # M2 = (a10+a11)b00
        # M3 = a00(b10-b11)
        # M4 = a11(b01-b00)
        # M5 = (a00+a01)b11
        # M6 = (a10-a00)(b00+b10)
        # M7 = (a01-a11)(b01+b11)
        # C: c00 = M1-M4+M6+M7, c01 = M3+M5, c10 = M2+M4, c11 = M1-M3+M5-M7

        # a-index: a00=0, a01=1, a10=2, a11=3
        # b-index: b00=0, b01=1, b10=2, b11=3
        # c-index: c00=0, c01=1, c10=2, c11=3

        factors = [
            # (U_entries, V_entries, W_entries)  as dict {index: coeff}
            # M1 = (a00+a11)(b00+b11)
            ({0:1, 3:1}, {0:1, 3:1}, {0:1, 3:-1}),
            # M2 = (a10+a11)b00
            ({2:1, 3:1}, {0:1}, {2:1}),
            # M3 = a00(b10-b11)
            ({0:1}, {2:1, 3:-1}, {1:1}),
            # M4 = a11(b01-b00)
            ({3:1}, {1:1, 0:-1}, {0:-1, 2:1}),
            # M5 = (a00+a01)b11
            ({0:1, 1:1}, {3:1}, {1:1, 3:-1}),
            # M6 = (a10-a00)(b00+b10)
            ({2:1, 0:-1}, {0:1, 2:1}, {0:1}),
            # M7 = (a01-a11)(b01+b11)
            ({1:1, 3:-1}, {1:1, 3:1}, {0:1, 3:-1}),
        ]
        for r, (ud, vd, wd) in enumerate(factors):
            for idx, val in ud.items():
                U[idx, r] += val
            for idx, val in vd.items():
                V[idx, r] += val
            for idx, val in wd.items():
                W[idx, r] += val
        return U, V, W

    # Verify Strassen core
    U2, V2, W2 = strassen_2x2x2()
    T2 = np.einsum('ar,br,cr->abc', U2, V2, W2)
    # Check: T2[a,b,c] should be 1 iff a=i*2+j, b=j*2+k, c=k*2+i
    ok2 = True
    for i in range(2):
        for j in range(2):
            for k in range(2):
                a, b, c = i*2+j, j*2+k, k*2+i
                if abs(T2[a,b,c] - 1.0) > 1e-9:
                    ok2 = False
    if not ok2:
        # Fallback: schoolbook for 2×2×2
        U2 = np.zeros((4,4)); V2 = np.zeros((4,4)); W2 = np.zeros((4,4))
        for col,(a,b,c) in enumerate([(i*2+j,j*2+k,k*2+i) for i in range(2) for j in range(2) for k in range(2)]):
            U2[a,col]=1; V2[b,col]=1; W2[c,col]=1

    # ── Build 4×4×8 decomposition using 2×2 Strassen blocks ──────────────
    # Partition: rows 0-3 → block-rows r ∈ {0,1}, within-row s ∈ {0,1}
    #            cols 0-7 → block-cols c ∈ {0,1,2,3}, within-col d ∈ {0,1}
    # 4 blocks of (2×2)·(2×2) → rank 7 each → total 28
    # But we have 4×4 × 4×8: 4 block-rows × 4 block-cols of 2×2
    # Actually 4 rows = 2 block-rows, 8 cols = 4 block-cols → 8 sub-blocks
    # Each sub-block is 2×2 · 2×2 → rank 7 → total 56.
    #
    # Hmm, that's not better than schoolbook 144.  Let's think again.
    #
    # 4×4·4×8:  2×2 grid of 2×2 blocks for A,  2×4 grid of 2×2 blocks for B
    # → 4 sub-products, each 2×2·2×2 → rank 7 each → total 28.  Yes!
    #
    # A = [[A00, A01],   B = [[B00, B01, B02, B03],
    #      [A10, A11]]      [B10, B11, B12, B13]]
    # C = A·B = [[C00, C01, C02, C03],
    #            [C10, C11, C12, C13]]
    # where each Cij is 2×2.
    #
    # C00 = A00·B00 + A01·B10
    # C01 = A00·B01 + A01·B11
    # etc.
    #
    # Strassen gives rank-7 for each 2×2·2×2 product.
    # But C00 = A00·B00 + A01·B10 requires 2 separate Strassen decompositions
    # (14 factors), unless we can combine.  They operate on different
    # sub-matrices so they're independent → 28 factors for row-0 of C.
    # Similarly 28 for row-1 → total 56.
    #
    # This is still an improvement over 144 but not great.  Let's try a
    # different partition.

    # ── Try: 4×4·4×9 with direct 2×2 Strassen on padded 4×4·4×8 + 4 extra ─
    # Pad B to 4×8 (zero out cols 8-11... wait, B is 4×9, pad to 4×12? No.)
    # Let's pad B from 4×9 to 4×12 (3 zero cols), C from 4×9 to 4×12.
    # Then 4×4·4×12: 2×2 blocks for A, 2×6 blocks for B → 12 sub-products
    # of 2×2·2×2 → 12*7 = 84 rank.  Worse.
    #
    # Let's just use the ALS approach but with a better initialization.

    # ── Phase A: fast deterministic construction using known results ─────
    # For 4×4·4×4, the best known rank is 19 (Strassen's 19-term formula).
    # For 4×4·4×9, we can write B as [B4 | B5] where B4 is 4×4 and B5 is 4×5.
    # C = A·B4 + A·B5.
    # A·B4 has rank ≤ 19.
    # A·B5: B5 is 4×5.  Pad to 4×8, use 4×4·4×8 (rank 28) → A·B5 rank ≤ 28.
    # Total ≤ 19 + 28 = 47.
    #
    # Actually, let's just try to use the Strassen 19-term formula for 4×4×4
    # and add 4×5 handling.
    #
    # The Strassen 19-term formula for 4×4×4 is complex.  Let's use a
    # simpler approach: partition 4×4 into 2×2 blocks and use 2×2 Strassen.
    # 4×4·4×4 = 4 sub-products of 2×2·2×2 → 4*7 = 28 rank.
    #
    # For 4×4·4×9:
    #   Partition A into 2×2 blocks: A = [[A00,A01],[A10,A11]]
    #   Partition B (4×9) into 2×2 and 2×1 blocks:
    #     B = [[B00(2×2), B01(2×2), B02(2×2), B03(2×1)],
    #          [B10(2×2), B11(2×2), B12(2×2), B13(2×1)]]
    #   C (4×9) = [[C00,C01,C02,C03],[C10,C11,C12,C13]] where each Cij is 2×2 except C03,C13 are 2×1.
    #
    # For the 2×2·2×2 products: rank 7 each.  8 such products → 56.
    # For the 2×2·2×1 products: these are just 2×2·(2×1) = each output entry
    #   is a dot product of 2 elements.  We can do this with 2 rank-1 terms each.
    #   2 such products → 4 rank-1 terms.
    # Total: 56 + 4 = 60.
    #
    # This is an improvement over 144.  Let's construct it.

    U_blocks = []
    V_blocks = []
    W_blocks = []

    # 2×2 Strassen core
    U2, V2, W2 = strassen_2x2x2()

    for br in range(2):  # block-row of A
        for bc in range(4):  # block-col of B (0..3)
            # Sub-product: A[br] (2×2) · B[bc] (2×2 or 2×1)
            # Global indices:
            #   A rows: br*2, br*2+1
            #   A cols: bc*2, bc*2+1  (if bc<4, else bc*2 only for bc=4)
            #   B rows: same as A cols
            #   B cols: bc*2, bc*2+1 (if bc<4) or bc*2 (if bc=4, but we only have 9 cols)
            #   C rows: br*2, br*2+1
            #   C cols: bc*2, bc*2+1

            a_base = br * 2          # A row offset
            col_base = bc * 2        # A col / B row offset
            b_col_base = bc * 2      # B col offset

            if bc < 4:
                # 2×2 · 2×2
                b_col_size = 2
            else:
                b_col_size = 1  # only col 8

            # For 2×2·2×2, use Strassen (7 factors)
            # Map local 2×2 indices to global
            for r in range(7):
                u_row = np.zeros(U_rows)
                v_row = np.zeros(V_rows)
                w_row = np.zeros(W_rows)
                for local_a in range(4):
                    i, j = local_a // 2, local_a % 2
                    ga = (a_base + i) * m + (col_base + j)
                    u_row[ga] = U2[local_a, r]
                for local_b in range(4):
                    j, k = local_b // 2, local_b % 2
                    gb = (col_base + j) * p + (b_col_base + k)
                    if gb < V_rows:
                        v_row[gb] = V2[local_b, r]
                for local_c in range(4):
                    k, i = local_c // 2, local_c % 2
                    gc = (b_col_base + k) * n + (a_base + i)
                    if gc < W_rows:
                        w_row[gc] = W2[local_c, r]
                U_blocks.append(u_row)
                V_blocks.append(v_row)
                W_blocks.append(w_row)

            if bc >= 4:
                # 2×2 · 2×1: only 1 col of B
                # Use 2 rank-1 terms
                for t in range(2):
                    u_row = np.zeros(U_rows)
                    v_row = np.zeros(V_rows)
                    w_row = np.zeros(W_rows)
                    # A row t of the 2×2 block
                    for j in range(2):
                        ga = (a_base + t) * m + (col_base + j)
                        u_row[ga] = 1.0
                    gb = (col_base + t) * p + b_col_base
                    if gb < V_rows:
                        v_row[gb] = 1.0
                    for i in range(2):
                        gc = b_col_base * n + (a_base + i)
                        if gc < W_rows:
                            w_row[gc] = 1.0
                    U_blocks.append(u_row)
                    V_blocks.append(v_row)
                    W_blocks.append(w_row)

    U_A = np.column_stack(U_blocks) if U_blocks else np.zeros((U_rows, 1))
    V_A = np.column_stack(V_blocks) if V_blocks else np.zeros((V_rows, 1))
    W_A = np.column_stack(W_blocks) if W_blocks else np.zeros((W_rows, 1))

    rank_A = U_A.shape[1]
    print(f"Phase A rank: {rank_A}")

    # Verify
    if exact_check(U_A, V_A, W_A):
        print("Phase A: exact!")
        best_U, best_V, best_W = U_A, V_A, W_A
        best_rank = rank_A
    else:
        print("Phase A: not exact, using schoolbook")
        best_U, best_V, best_W = U_school, V_school, W_school
        best_rank = rank_school

    # ── Phase B: ALS search for lower rank ───────────────────────────────
    T_U_flat = T_U.reshape(U_rows, V_rows * W_rows)
    T_V = np.zeros((V_rows, U_rows, W_rows))
    T_W = np.zeros((W_rows, U_rows, V_rows))
    for (a, b, c) in triples:
        T_V[b, a, c] = 1.0
        T_W[c, a, b] = 1.0
    T_V_flat = T_V.reshape(V_rows, U_rows * W_rows)
    T_W_flat = T_W.reshape(W_rows, U_rows * V_rows)

    def als_decompose(target_rank, max_iters=60, tol=1e-10, seed=42):
        if time.time() > deadline - 15:
            return None
        rng = np.random.RandomState(seed)
        U = rng.randn(U_rows, target_rank) * 0.01
        V = rng.randn(V_rows, target_rank) * 0.01
        W = rng.randn(W_rows, target_rank) * 0.01
        prev_err = float('inf')

        for it in range(max_iters):
            if time.time() > deadline - 15:
                break
            # Update U
            B_t = np.einsum('br,cr->bcr', V, W)
            B = B_t.transpose(2, 0, 1).reshape(target_rank, V_rows * W_rows)
            G = B @ B.T + 1e-10 * np.eye(target_rank)
            try:
                G_inv = np.linalg.inv(G)
            except:
                G_inv = np.linalg.pinv(G)
            U = (G_inv @ (B @ T_U_flat.T)).T

            # Update V
            C_t = np.einsum('ar,cr->acr', U, W)
            C = C_t.transpose(2, 0, 1).reshape(target_rank, U_rows * W_rows)
            Gv = C @ C.T + 1e-10 * np.eye(target_rank)
            try:
                Gv_inv = np.linalg.inv(Gv)
            except:
                Gv_inv = np.linalg.pinv(Gv)
            V = (Gv_inv @ (C @ T_V_flat.T)).T

            # Update W
            D_t = np.einsum('ar,br->abr', U, V)
            D = D_t.transpose(2, 0, 1).reshape(target_rank, U_rows * V_rows)
            Gw = D @ D.T + 1e-10 * np.eye(target_rank)
            try:
                Gw_inv = np.linalg.inv(Gw)
            except:
                Gw_inv = np.linalg.pinv(Gw)
            W = (Gw_inv @ (D @ T_W_flat.T)).T

            if it % 10 == 9:
                err = 0.0
                for (a, b, c) in triples:
                    val = np.dot(U[a, :] * V[b, :], W[c, :])
                    err += (val - 1.0) ** 2
                rng_c = np.random.RandomState(it)
                for _ in range(200):
                    a = rng_c.randint(0, U_rows)
                    b = rng_c.randint(0, V_rows)
                    c = rng_c.randint(0, W_rows)
                    if (a, b, c) in triple_set:
                        continue
                    val = np.dot(U[a, :] * V[b, :], W[c, :])
                    err += val ** 2
                if err < tol:
                    break
                if abs(prev_err - err) < 1e-14:
                    break
                prev_err = err
        return U, V, W

    def try_round(U, V, W, scale=2):
        U_r = np.round(U * scale) / scale
        V_r = np.round(V * scale) / scale
        W_r = np.round(W * scale) / scale
        if exact_check(U_r, V_r, W_r):
            return U_r, V_r, W_r
        return None

    # Try ranks from best_rank-1 down to 30
    for target_rank in range(min(best_rank - 1, 60), 29, -1):
        if time.time() > deadline - 25:
            break
        found = False
        for seed in range(4):
            if time.time() > deadline - 25:
                break
            result = als_decompose(target_rank, max_iters=50, seed=seed)
            if result is None:
                continue
            U, V, W = result
            if exact_check(U, V, W, tol=1e-6):
                rounded = try_round(U, V, W)
                if rounded is not None:
                    best_U, best_V, best_W = rounded
                    best_rank = target_rank
                    found = True
                    break
        if found:
            break

    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
# EVOLVE-BLOCK-END
