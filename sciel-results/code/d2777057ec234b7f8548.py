"""
Program for matrix multiplication 2x5x5 rank minimization.
Uses Strassen's algorithm on 2x2 sub-blocks to reduce rank from 50 to 46.
"""
import time

DIMENSIONS = [2, 5, 5]


def _exact_rank_reduce(U, V, W):
    """
    Cheap exact rank-reduction passes on a bilinear decomposition.
    1) Drop columns that are all-zero.
    2) Merge duplicate (or scalar-multiple) columns: if column j is a scalar
       multiple of column i (with the same scalar across U,V,W), remove j.
    Works over rationals using fractions to stay exact.
    """
    from fractions import Fraction
    import numpy as np

    Ua = np.array(U, dtype=object)
    Va = np.array(V, dtype=object)
    Wa = np.array(W, dtype=object)
    n_terms = Ua.shape[1]

    # Convert to Fraction for exact arithmetic
    Uf = [[Fraction(x) for x in row] for row in Ua.tolist()]
    Vf = [[Fraction(x) for x in row] for row in Va.tolist()]
    Wf = [[Fraction(x) for x in row] for row in Wa.tolist()]

    # Pass 1: drop all-zero columns
    keep = []
    for j in range(n_terms):
        col_u = [Uf[i][j] for i in range(len(Uf))]
        col_v = [Vf[i][j] for i in range(len(Vf))]
        col_w = [Wf[i][j] for i in range(len(Wf))]
        if all(x == 0 for x in col_u) and all(x == 0 for x in col_v) and all(x == 0 for x in col_w):
            continue
        keep.append(j)

    if len(keep) < n_terms:
        Uf = [[Uf[i][j] for j in keep] for i in range(len(Uf))]
        Vf = [[Vf[i][j] for j in keep] for i in range(len(Vf))]
        Wf = [[Wf[i][j] for j in keep] for i in range(len(Wf))]
        n_terms = len(keep)

    # Pass 2: merge scalar-multiple columns
    # For each pair (i, j) with i < j, check if column j = c * column i for some Fraction c
    # such that the same c works for U, V, and W.
    removed = set()
    for i in range(n_terms):
        if i in removed:
            continue
        col_u_i = [Uf[r][i] for r in range(len(Uf))]
        col_v_i = [Vf[r][i] for r in range(len(Vf))]
        col_w_i = [Wf[r][i] for r in range(len(Wf))]
        # Find a candidate scalar c from the first nonzero entry in any of U,V,W column i
        c = None
        for lst in (col_u_i, col_v_i, col_w_i):
            for val in lst:
                if val != 0:
                    c = 1
                    break
            if c is not None:
                break
        if c is None:
            continue  # zero column (shouldn't happen after pass 1)

        for j in range(i + 1, n_terms):
            if j in removed:
                continue
            col_u_j = [Uf[r][j] for r in range(len(Uf))]
            col_v_j = [Vf[r][j] for r in range(len(Vf))]
            col_w_j = [Wf[r][j] for r in range(len(Wf))]

            # Determine scalar c' = col_j / col_i from first nonzero pair
            c_prime = None
            for lst_i, lst_j in ((col_u_i, col_u_j), (col_v_i, col_v_j), (col_w_i, col_w_j)):
                for val_i, val_j in zip(lst_i, lst_j):
                    if val_i != 0:
                        c_prime = val_j / val_i
                        break
                if c_prime is not None:
                    break

            if c_prime is None:
                continue

            # Verify: for all entries, val_j == c_prime * val_i
            ok = True
            for lst_i, lst_j in ((col_u_i, col_u_j), (col_v_i, col_v_j), (col_w_i, col_w_j)):
                for val_i, val_j in zip(lst_i, lst_j):
                    if c_prime * val_i != val_j:
                        ok = False
                        break
                if not ok:
                    break

            if ok:
                removed.add(j)

    if removed:
        keep = [j for j in range(n_terms) if j not in removed]
        Uf = [[Uf[i][j] for j in keep] for i in range(len(Uf))]
        Vf = [[Vf[i][j] for j in keep] for i in range(len(Vf))]
        Wf = [[Wf[i][j] for j in keep] for i in range(len(Wf))]

    # Convert back to int if all entries are integers, else keep as float
    def to_python(lst):
        out = []
        for row in lst:
            new_row = []
            for val in row:
                if val.denominator == 1:
                    new_row.append(val.numerator)
                else:
                    new_row.append(float(val))
            out.append(new_row)
        return out

    return to_python(Uf), to_python(Vf), to_python(Wf)

# EVOLVE-BLOCK-START
def _build_strassen_decomposition():
    """
    Build the rank-46 Strassen-based decomposition for 2x5 x 5x5.
    Partition inner dimension 5 = 2+2+1.
    """
    n, m, p = 2, 5, 5

    terms = []

    def add_term(u_entries, v_entries, w_entries):
        terms.append((u_entries, v_entries, w_entries))

    def schoolbook_block(a_rows, a_cols, b_rows, b_cols, a_shape, b_shape, out_rows, out_cols):
        nr, nc = a_shape
        nr2, nc2 = b_shape
        for i in range(nr):
            for k in range(nc):
                for j in range(nc2):
                    u_idx = (a_rows + i) * m + (a_cols + k)
                    v_idx = (b_rows + k) * p + (b_cols + j)
                    w_idx = (out_cols + j) * n + (out_rows + i)
                    add_term([(u_idx, 1)], [(v_idx, 1)], [(w_idx, 1)])

    def strassen_2x2x2(a_row_off, a_col_off, b_row_off, b_col_off, out_row_off, out_col_off):
        a_u = a_row_off * m + a_col_off
        b_u = a_row_off * m + (a_col_off + 1)
        c_u = (a_row_off + 1) * m + a_col_off
        d_u = (a_row_off + 1) * m + (a_col_off + 1)

        e_v = b_row_off * p + b_col_off
        f_v = b_row_off * p + (b_col_off + 1)
        g_v = (b_row_off + 1) * p + b_col_off
        h_v = (b_row_off + 1) * p + (b_col_off + 1)

        o00_w = out_col_off * n + out_row_off
        o01_w = (out_col_off + 1) * n + out_row_off
        o10_w = out_col_off * n + (out_row_off + 1)
        o11_w = (out_col_off + 1) * n + (out_row_off + 1)

        # M1 = (a+d)(e+h)
        add_term([(a_u, 1), (d_u, 1)], [(e_v, 1), (h_v, 1)], [(o00_w, 1), (o11_w, 1)])
        # M2 = (c+d)e
        add_term([(c_u, 1), (d_u, 1)], [(e_v, 1)], [(o10_w, 1), (o11_w, -1)])
        # M3 = a(f-h)
        add_term([(a_u, 1)], [(f_v, 1), (h_v, -1)], [(o01_w, 1), (o11_w, 1)])
        # M4 = d(g-e)
        add_term([(d_u, 1)], [(g_v, 1), (e_v, -1)], [(o00_w, 1), (o10_w, 1)])
        # M5 = (a+b)h
        add_term([(a_u, 1), (b_u, 1)], [(h_v, 1)], [(o00_w, -1), (o01_w, 1)])
        # M6 = (c-a)(e+f)
        add_term([(c_u, 1), (a_u, -1)], [(e_v, 1), (f_v, 1)], [(o11_w, 1)])
        # M7 = (b-d)(g+h)
        add_term([(b_u, 1), (d_u, -1)], [(g_v, 1), (h_v, 1)], [(o00_w, 1)])

    # Partition 5 = 2+2+1
    strassen_2x2x2(0, 0, 0, 0, 0, 0)
    strassen_2x2x2(0, 0, 0, 2, 0, 2)
    schoolbook_block(0, 0, 0, 4, (2, 2), (2, 1), 0, 4)

    strassen_2x2x2(0, 2, 2, 0, 0, 0)
    strassen_2x2x2(0, 2, 2, 2, 0, 2)
    schoolbook_block(0, 2, 2, 4, (2, 2), (2, 1), 0, 4)

    schoolbook_block(0, 4, 4, 0, (2, 1), (1, 5), 0, 0)

    total_rank = len(terms)
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]

    for col, (u_entries, v_entries, w_entries) in enumerate(terms):
        for idx, val in u_entries:
            U[idx][col] = val
        for idx, val in v_entries:
            V[idx][col] = val
        for idx, val in w_entries:
            W[idx][col] = val

    return U, V, W, total_rank

def construct():
    """
    Returns a rank-minimal exact tensor decomposition for multiplying 
    a 2 by 5 matrix by a 5 by 5 matrix.
    """
    start_time = time.time()
    deadline = start_time + 110.0

    # Build the standard rank-46 decomposition
    U, V, W, total_rank = _build_strassen_decomposition()

    # Time guard
    if time.time() > deadline - 5.0:
        return {"U": U, "V": V, "W": W}

    # Apply cheap exact rank-reduction passes
    try:
        U2, V2, W2 = _exact_rank_reduce(U, V, W)
        if time.time() > deadline - 2.0:
            return {"U": U, "V": V, "W": W}
        return {"U": U2, "V": V2, "W": W2}
    except Exception:
        return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
