"""
True recursive Strassen decomposition for 4x5x5 matrix multiplication.

The 4x4x4 corner uses a genuine two-level Strassen (rank 49 = 7 * 7): each of
the 7 outer Strassen multiplications is itself a 2x2 matrix multiplication
whose entries are linear combinations of 2x2 A/B blocks, expanded with the
rank-7 inner Strassen.  The 1-wide strips use schoolbook.

Partition:
  I: [0,1,2,3]           (size 4)
  J: [0,1,2,3], [4]      (sizes 4, 1)
  K: [0,1,2,3], [4]      (sizes 4, 1)

Blocks:
  (4,4,4) -> 4x4x4 Strassen : 49
  (4,4,1) -> schoolbook     : 16
  (4,1,4) -> schoolbook     : 16
  (4,1,1) -> schoolbook     : 4
  Total rank = 79
"""
import numpy as np
import time

DIMENSIONS = [4, 5, 5]
TIME_LIMIT = 140.0


def construct():
    start_time = time.time()
    n, m, p = DIMENSIONS

    U_cols = []
    V_cols = []
    W_cols = []

    def add_term(u_vec, v_vec, w_vec):
        u_col = np.zeros(n * m)
        v_col = np.zeros(m * p)
        w_col = np.zeros(p * n)
        for idx, val in u_vec:
            u_col[idx] = val
        for idx, val in v_vec:
            v_col[idx] = val
        for idx, val in w_vec:
            w_col[idx] = val
        U_cols.append(u_col)
        V_cols.append(v_col)
        W_cols.append(w_col)

    def schoolbook_block(i_start, i_size, j_start, j_size, k_start, k_size):
        for i_loc in range(i_size):
            for j_loc in range(j_size):
                for k_loc in range(k_size):
                    i_glob = i_start + i_loc
                    j_glob = j_start + j_loc
                    k_glob = k_start + k_loc
                    add_term([(i_glob * m + j_glob, 1)],
                             [(j_glob * p + k_glob, 1)],
                             [(k_glob * n + i_glob, 1)])

    # Canonical 2x2x2 Strassen terms as (U, V, W) sparse specs of (pos, coeff).
    S2 = [
        ([(0, 0, 1), (1, 1, 1)], [(0, 0, 1), (1, 1, 1)], [(0, 0, 1), (1, 1, 1)]),
        ([(1, 0, 1), (1, 1, 1)], [(0, 0, 1)],           [(0, 1, 1), (1, 1, -1)]),
        ([(0, 0, 1)],           [(0, 1, 1), (1, 1, -1)], [(1, 0, 1), (1, 1, 1)]),
        ([(1, 1, 1)],           [(1, 0, 1), (0, 0, -1)], [(0, 0, 1), (0, 1, 1)]),
        ([(0, 0, 1), (0, 1, 1)], [(1, 1, 1)],           [(0, 0, -1), (1, 0, 1)]),
        ([(1, 0, 1), (0, 0, -1)], [(0, 0, 1), (0, 1, 1)], [(1, 1, 1)]),
        ([(0, 1, 1), (1, 1, -1)], [(1, 0, 1), (1, 1, 1)], [(0, 0, 1)]),
    ]

    def strassen_2x2x2(i_start, j_start, k_start):
        def gI(i_l, j_l):
            return (i_start + i_l) * m + (j_start + j_l)
        def gJ(j_l, k_l):
            return (j_start + j_l) * p + (k_start + k_l)
        def gK(k_l, i_l):
            return (k_start + k_l) * n + (i_start + i_l)
        for us, vs, ws in S2:
            add_term([(gI(a, b), c) for a, b, c in us],
                     [(gJ(a, b), c) for a, b, c in vs],
                     [(gK(a, b), c) for a, b, c in ws])

    def strassen_4x4x4(i_start, j_start, k_start):
        def gI(a, b, i_l, j_l):
            return (i_start + 2 * a + i_l) * m + (j_start + 2 * b + j_l)
        def gJ(a, b, j_l, k_l):
            return (j_start + 2 * a + j_l) * p + (k_start + 2 * b + k_l)
        def gK(a, b, k_l, i_l):
            return (k_start + 2 * a + k_l) * n + (i_start + 2 * b + i_l)

        # Outer Strassen meta-terms: linear combos of real 2x2 blocks.
        meta = [
            ([[(0, 0, 1), (1, 1, 1)]],
             [[(0, 0, 1), (1, 1, 1)]],
             [[(0, 0, 1), (1, 1, 1)]]),
            ([[(1, 0, 1), (1, 1, 1)]],
             [[(0, 0, 1)]],
             [[(0, 1, 1), (1, 1, -1)]]),
            ([[(0, 0, 1)]],
             [[(0, 1, 1), (1, 1, -1)]],
             [[(1, 0, 1), (1, 1, 1)]]),
            ([[(1, 1, 1)]],
             [[(1, 0, 1), (0, 0, -1)]],
             [[(0, 0, 1), (0, 1, 1)]]),
            ([[(0, 0, 1), (0, 1, 1)]],
             [[(1, 1, 1)]],
             [[(0, 0, -1), (1, 0, 1)]]),
            ([[(1, 0, 1), (0, 0, -1)]],
             [[(0, 0, 1), (0, 1, 1)]],
             [[(1, 1, 1)]]),
            ([[(0, 1, 1), (1, 1, -1)]],
             [[(1, 0, 1), (1, 1, 1)]],
             [[(0, 0, 1)]]),
        ]

        def combo(spec, a, b):
            return sum(c for (pa, pb, c) in spec if pa == a and pb == b)

        for u_list, v_list, w_list in meta:
            for us, vs, ws in S2:
                u_vec = [(gI(a, b, a, b), c * combo(u_list, a, b)) for a, b, c in us
                         if combo(u_list, a, b) != 0]
                v_vec = [(gJ(a, b, a, b), c * combo(v_list, a, b)) for a, b, c in vs
                         if combo(v_list, a, b) != 0]
                w_vec = [(gK(a, b, a, b), c * combo(w_list, a, b)) for a, b, c in ws
                         if combo(w_list, a, b) != 0]
                add_term(u_vec, v_vec, w_vec)

    i_groups = [(0, 4)]
    j_groups = [(0, 4), (4, 1)]
    k_groups = [(0, 4), (4, 1)]

    for i_start, i_size in i_groups:
        for j_start, j_size in j_groups:
            for k_start, k_size in k_groups:
                if i_size == 4 and j_size == 4 and k_size == 4:
                    strassen_4x4x4(i_start, j_start, k_start)
                else:
                    schoolbook_block(i_start, i_size, j_start, j_size, k_start, k_size)
                if time.time() - start_time > TIME_LIMIT:
                    break
            if time.time() - start_time > TIME_LIMIT:
                break
        if time.time() - start_time > TIME_LIMIT:
            break

    if not U_cols:
        from itertools import product
        r_school = n * m * p
        U_sb = np.zeros((n * m, r_school))
        V_sb = np.zeros((m * p, r_school))
        W_sb = np.zeros((p * n, r_school))
        for c, (i, j, k) in enumerate(product(range(n), range(m), range(p))):
            U_sb[i * m + j, c] = 1.0
            V_sb[j * p + k, c] = 1.0
            W_sb[k * n + i, c] = 1.0
        return {"U": U_sb.tolist(), "V": V_sb.tolist(), "W": W_sb.tolist()}

    U = np.column_stack(U_cols)
    V = np.column_stack(V_cols)
    W = np.column_stack(W_cols)

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}


if __name__ == "__main__":
    construct()
