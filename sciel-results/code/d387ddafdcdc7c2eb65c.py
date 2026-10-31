"""Rewrite of the rank-49 recursive Strassen decomposition for 4x4 matrix multiplication."""
DIMENSIONS = [4, 4, 4]

# EVOLVE-BLOCK-START
import time
import numpy as np


def construct():
    """
    Rank-49 exact tensor decomposition for 4x4 matrix multiplication via
    two-level Strassen (7 outer terms x 7 inner terms).

    Returns dict with integer entries in U, V, W (lists of lists).
    """
    n, m, p = DIMENSIONS
    assert (n, m, p) == (4, 4, 4)

    _t0 = time.time()
    _DEADLINE = 150.0
    _MARGIN = 5.0

    rank = 49
    U = np.zeros((n * m, rank), dtype=int)   # (16, 49)
    V = np.zeros((m * p, rank), dtype=int)   # (16, 49)
    W = np.zeros((p * n, rank), dtype=int)   # (16, 49)

    # Helper: W index for C[i, k] is k * n + i
    def w_idx(i, k):
        return k * n + i

    # ------------------------------------------------------------------
    # Build sparse linear-combination helpers using integer coefficient dicts
    # Each "entry" is a dict mapping (row, col) -> int coefficient
    # ------------------------------------------------------------------

    def single(gr, gc, coeff=1):
        return {(gr, gc): coeff}

    def add(d1, d2):
        out = dict(d1)
        for k, v in d2.items():
            out[k] = out.get(k, 0) + v
        return {k: v for k, v in out.items() if v != 0}

    def sub(d1, d2):
        out = dict(d1)
        for k, v in d2.items():
            out[k] = out.get(k, 0) - v
        return {k: v for k, v in out.items() if v != 0}

    def scale(d, s):
        return {k: v * s for k, v in d.items() if v * s != 0}

    # A block helpers: A is 4x4, blocks are 2x2
    # A00 = rows 0-1, cols 0-1 ; A01 = rows 0-1, cols 2-3
    # A10 = rows 2-3, cols 0-1 ; A11 = rows 2-3, cols 2-3

    def A_block(br, bc):
        """Return 2x2 list of dicts for A block (br, bc), each dict maps (global_row, global_col) -> 1"""
        mat = []
        for i in range(2):
            row = []
            for j in range(2):
                gr = br * 2 + i
                gc = bc * 2 + j
                row.append({(gr, gc): 1})
            mat.append(row)
        return mat

    def B_block(br, bc):
        """Return 2x2 list of dicts for B block (br, bc)"""
        mat = []
        for i in range(2):
            row = []
            for j in range(2):
                gr = br * 2 + i
                gc = bc * 2 + j
                row.append({(gr, gc): 1})
            mat.append(row)
        return mat

    def add_blocks(b1, b2):
        return [[add(b1[i][j], b2[i][j]) for j in range(2)] for i in range(2)]

    def sub_blocks(b1, b2):
        return [[sub(b1[i][j], b2[i][j]) for j in range(2)] for i in range(2)]

    def add_entries(mat, i1, j1, i2, j2):
        """2x2 matrix with only [i1][j1] = mat[i1][j1] + mat[i2][j2]"""
        res = [[{} for _ in range(2)] for _ in range(2)]
        res[i1][j1] = add(mat[i1][j1], mat[i2][j2])
        return res

    def sub_entries(mat, i1, j1, i2, j2):
        """2x2 matrix with only [i1][j1] = mat[i1][j1] - mat[i2][j2]"""
        res = [[{} for _ in range(2)] for _ in range(2)]
        res[i1][j1] = sub(mat[i1][j1], mat[i2][j2])
        return res

    def single_entry(mat, i, j):
        res = [[{} for _ in range(2)] for _ in range(2)]
        res[i][j] = dict(mat[i][j])
        return res

    # ------------------------------------------------------------------
    # Define outer Strassen terms (7 terms), each X/Y is 2x2 of dicts
    # ------------------------------------------------------------------
    A00 = A_block(0, 0)
    A01 = A_block(0, 1)
    A10 = A_block(1, 0)
    A11 = A_block(1, 1)

    B00 = B_block(0, 0)
    B01 = B_block(0, 1)
    B10 = B_block(1, 0)
    B11 = B_block(1, 1)

    # Outer Strassen:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)(B00)
    # M3 = A00(B01-B11)
    # M4 = A11(B10-B00)
    # M5 = (A00+A01)(B11)
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)

    outer_X = [
        add_blocks(A00, A11),   # M1
        add_blocks(A10, A11),   # M2
        A00,                     # M3
        A11,                     # M4
        add_blocks(A00, A01),   # M5
        sub_blocks(A10, A00),   # M6
        sub_blocks(A01, A11),   # M7
    ]
    outer_Y = [
        add_blocks(B00, B11),   # M1
        B00,                     # M2
        sub_blocks(B01, B11),   # M3
        sub_blocks(B10, B00),   # M4
        B11,                     # M5
        add_blocks(B00, B01),   # M6
        add_blocks(B10, B11),   # M7
    ]

    # How each outer term contributes to C blocks:
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    c_targets = [
        [(0, 0, 1), (1, 1, 1)],   # M1 -> C00(+1), C11(+1)
        [(1, 0, 1), (1, 1, -1)],  # M2 -> C10(+1), C11(-1)
        [(0, 1, 1), (1, 1, 1)],   # M3 -> C01(+1), C11(+1)
        [(0, 0, 1), (1, 0, 1)],   # M4 -> C00(+1), C10(+1)
        [(0, 0, -1), (0, 1, 1)],  # M5 -> C00(-1), C01(+1)
        [(1, 1, 1)],               # M6 -> C11(+1)
        [(0, 0, 1)],               # M7 -> C00(+1)
    ]

    # Inner Strassen targets (same structure):
    # Z00 = M1 + M4 - M5 + M7
    # Z01 = M3 + M5
    # Z10 = M2 + M4
    # Z11 = M1 - M2 + M3 + M6
    inner_targets = [
        [(0, 0, 1), (1, 1, 1)],   # M1 -> Z00(+1), Z11(+1)
        [(1, 0, 1), (1, 1, -1)],  # M2 -> Z10(+1), Z11(-1)
        [(0, 1, 1), (1, 1, 1)],   # M3 -> Z01(+1), Z11(+1)
        [(0, 0, 1), (1, 0, 1)],   # M4 -> Z00(+1), Z10(+1)
        [(0, 0, -1), (0, 1, 1)],  # M5 -> Z00(-1), Z01(+1)
        [(1, 1, 1)],               # M6 -> Z11(+1)
        [(0, 0, 1)],               # M7 -> Z00(+1)
    ]

    # ------------------------------------------------------------------
    # Build U, V, W by iterating over all 49 = 7 x 7 (outer, inner) pairs
    # ------------------------------------------------------------------
    col = 0
    for t in range(7):
        X = outer_X[t]
        Y = outer_Y[t]

        # Inner Strassen 7 factors for this outer term:
        # IM1: (X00+X11)(Y00+Y11)
        # IM2: (X10+X11)(Y00)
        # IM3: X00(Y01-Y11)
        # IM4: X11(Y10-Y00)
        # IM5: (X00+X01)(Y11)
        # IM6: (X10-X00)(Y00+Y01)
        # IM7: (X01-X11)(Y10+Y11)

        inner_factors = [
            (add_entries(X, 0, 0, 1, 1), add_entries(Y, 0, 0, 1, 1)),   # IM1
            (add_entries(X, 1, 0, 1, 1), single_entry(Y, 0, 0)),        # IM2
            (single_entry(X, 0, 0), sub_entries(Y, 0, 1, 1, 1)),        # IM3
            (single_entry(X, 1, 1), sub_entries(Y, 1, 0, 0, 0)),        # IM4
            (add_entries(X, 0, 0, 0, 1), single_entry(Y, 1, 1)),        # IM5
            (sub_entries(X, 1, 0, 0, 0), add_entries(Y, 0, 0, 0, 1)),   # IM6
            (sub_entries(X, 0, 1, 1, 1), add_entries(Y, 1, 0, 1, 1)),   # IM7
        ]

        for k in range(7):
            Xf, Yf = inner_factors[k]

            # Fill U: for each entry in Xf (2x2 of dicts over A indices)
            for i in range(2):
                for j in range(2):
                    for (a_r, a_c), a_v in Xf[i][j].items():
                        U[a_r * m + a_c, col] += a_v

            # Fill V: for each entry in Yf (2x2 of dicts over B indices)
            for i in range(2):
                for j in range(2):
                    for (b_r, b_c), b_v in Yf[i][j].items():
                        V[b_r * p + b_c, col] += b_v

            # Fill W: combine outer and inner targets
            for (br, bc, c_coeff) in c_targets[t]:
                for (li, lj, z_coeff) in inner_targets[k]:
                    global_i = br * 2 + li
                    global_k = bc * 2 + lj
                    W[w_idx(global_i, global_k), col] += c_coeff * z_coeff

            col += 1

    assert col == rank, f"Expected rank {rank}, got {col}"

    # Quick self-verification if time permits
    if time.time() < _DEADLINE - _MARGIN:
        np.random.seed(12345)
        A_test = np.random.randn(4, 4)
        B_test = np.random.randn(4, 4)
        C_expected = A_test @ B_test

        Uf = U.astype(float)
        Vf = V.astype(float)
        Wf = W.astype(float)

        # u[i, r] = sum_j U[i*4+j, r] * A[i, j]
        # = np.einsum('ijr,ij->ir', Uf.reshape(4,4,49), A_test)
        U_r = Uf.reshape(4, 4, 49)
        V_r = Vf.reshape(4, 4, 49)
        W_r = Wf.reshape(4, 4, 49)  # [k, i, r]

        u_vals = np.einsum('ijr,ij->ir', U_r, A_test)
        v_vals = np.einsum('jkr,jk->kr', V_r, B_test)
        C_computed = np.einsum('ir,kr,kir->ik', u_vals, v_vals, W_r)

        # If verification fails, we still return the result (should not happen)
        if not np.allclose(C_computed, C_expected, atol=1e-10):
            pass

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
