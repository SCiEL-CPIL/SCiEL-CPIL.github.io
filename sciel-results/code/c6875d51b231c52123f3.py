import numpy as np
import time

DIMENSIONS = [4, 5, 5]

def construct():
    n, m, p = DIMENSIONS
    n_m = n * m
    m_p = m * p
    p_n = p * n

    U_cols = []
    V_cols = []
    W_cols = []

    def add_term(u_vec, v_vec, w_vec):
        u_col = [0.0] * n_m
        v_col = [0.0] * m_p
        w_col = [0.0] * p_n
        for idx, val in u_vec:
            u_col[idx] = val
        for idx, val in v_vec:
            v_col[idx] = val
        for idx, val in w_vec:
            w_col[idx] = val
        U_cols.append(u_col)
        V_cols.append(v_col)
        W_cols.append(w_col)

    def schoolbook_block(i_off, j_off, k_off, n_sz, m_sz, p_sz):
        for ii in range(n_sz):
            for jj in range(m_sz):
                for kk in range(p_sz):
                    i_g = i_off + ii
                    j_g = j_off + jj
                    k_g = k_off + kk
                    add_term([(i_g * m + j_g, 1.0)],
                             [(j_g * p + k_g, 1.0)],
                             [(k_g * n + i_g, 1.0)])

    def strassen_2x2x2(i_off, j_off, k_off):
        def gI(a, b): return (i_off + a) * m + (j_off + b)
        def gJ(b, c): return (j_off + b) * p + (k_off + c)
        def gK(c, a): return (k_off + c) * n + (i_off + a)

        # M1 = (A00+A11)(B00+B11) -> C00 += M1, C11 += M1
        add_term([(gI(0,0), 1.0), (gI(1,1), 1.0)],
                 [(gJ(0,0), 1.0), (gJ(1,1), 1.0)],
                 [(gK(0,0), 1.0), (gK(1,1), 1.0)])

        # M2 = (A10+A11)B00 -> C10 += M2, C11 -= M2
        add_term([(gI(1,0), 1.0), (gI(1,1), 1.0)],
                 [(gJ(0,0), 1.0)],
                 [(gK(0,1), 1.0), (gK(1,1), -1.0)])

        # M3 = A00(B01-B11) -> C01 += M3, C11 += M3
        add_term([(gI(0,0), 1.0)],
                 [(gJ(0,1), 1.0), (gJ(1,1), -1.0)],
                 [(gK(1,0), 1.0), (gK(1,1), 1.0)])

        # M4 = A11(B10-B00) -> C00 += M4, C10 += M4
        add_term([(gI(1,1), 1.0)],
                 [(gJ(1,0), 1.0), (gJ(0,0), -1.0)],
                 [(gK(0,0), 1.0), (gK(0,1), 1.0)])

        # M5 = (A00+A01)B11 -> C00 -= M5, C01 += M5
        add_term([(gI(0,0), 1.0), (gI(0,1), 1.0)],
                 [(gJ(1,1), 1.0)],
                 [(gK(0,0), -1.0), (gK(1,0), 1.0)])

        # M6 = (A10-A00)(B00+B01) -> C11 += M6
        add_term([(gI(1,0), 1.0), (gI(0,0), -1.0)],
                 [(gJ(0,0), 1.0), (gJ(0,1), 1.0)],
                 [(gK(1,1), 1.0)])

        # M7 = (A01-A11)(B10+B11) -> C00 += M7
        add_term([(gI(0,1), 1.0), (gI(1,1), -1.0)],
                 [(gJ(1,0), 1.0), (gJ(1,1), 1.0)],
                 [(gK(0,0), 1.0)])

    # Partition: i=[0,1],[2,3]; j=[0,1],[2,3],[4]; k=[0,1],[2,3],[4]
    i_groups = [(0, 2), (2, 2)]
    j_groups = [(0, 2), (2, 2), (4, 1)]
    k_groups = [(0, 2), (2, 2), (4, 1)]

    for i_start, i_size in i_groups:
        for j_start, j_size in j_groups:
            for k_start, k_size in k_groups:
                if i_size == 2 and j_size == 2 and k_size == 2:
                    strassen_2x2x2(i_start, j_start, k_start)
                else:
                    schoolbook_block(i_start, j_start, k_start, i_size, j_size, k_size)

    # Verify the decomposition is correct
    U_np = np.array(U_cols, dtype=float).T  # shape (n_m, rank)
    V_np = np.array(V_cols, dtype=float).T  # shape (m_p, rank)
    W_np = np.array(W_cols, dtype=float).T  # shape (p_n, rank)

    rank = len(U_cols)
    ok = True
    for i in range(n):
        for j in range(m):
            for k in range(p):
                for r in range(n):
                    expected = 1.0 if (i == r and j == k) else 0.0
                    s = 0.0
                    for c in range(rank):
                        s += U_np[i*m+j, c] * V_np[j*p+k, c] * W_np[k*n+r, c]
                    if abs(s - expected) > 1e-9:
                        ok = False
                        break
                if not ok:
                    break
            if not ok:
                break
        if not ok:
            break

    if not ok:
        # Fallback to schoolbook
        U_cols = []
        V_cols = []
        W_cols = []
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    add_term([(i*m+j, 1.0)], [(j*p+k, 1.0)], [(k*n+i, 1.0)])
        U_np = np.array(U_cols, dtype=float).T
        V_np = np.array(V_cols, dtype=float).T
        W_np = np.array(W_cols, dtype=float).T

    return {
        "U": U_np.tolist(),
        "V": V_np.tolist(),
        "W": W_np.tolist()
    }
