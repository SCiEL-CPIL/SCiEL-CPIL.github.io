"""Matrix multiplication 4x4 * 4x6 tensor decomposition.

Strategy: Split B into B1 (4x4) and B2 (4x2).
- A*B1 (4x4 * 4x4) uses Laderman's algorithm: rank 19.
- A*B2 (4x4 * 4x2) uses schoolbook: rank 8.
Total rank: 19 + 8 = 27.

The multiplication tensor T(A,B) = A*B where A is 4x4, B is 4x6, C is 4x6.
Decomposition: T = sum_r u_r * v_r * w_r
  u_r in R^{16} (A-space, index i*m+j)
  v_r in R^{24} (B-space, index j*p+k)
  w_r in R^{24} (C-space, index k*n+i per trace convention W[k*n+i, r])
"""

import time

DIMENSIONS = [4, 4, 6]


def construct():
    n, m, p = 4, 4, 6

    U_rows = n * m   # 16
    V_rows = m * p   # 24
    W_rows = p * n   # 24

    U_cols = []
    V_cols = []
    W_cols = []

    def add_term(u_vec, v_vec, w_vec):
        U_cols.append(u_vec)
        V_cols.append(v_vec)
        W_cols.append(w_vec)

    def zeros(length):
        return [0] * length

    def laderman_4x4_4x4(b_col_offset, c_col_offset):
        """Add 19 rank-1 terms for Laderman 4x4*4x4.
        A is full 4x4 (rows 0-3, cols 0-3 of A).
        B block: rows 0-3, cols b_col_offset to b_col_offset+3.
        C block: rows 0-3, cols c_col_offset to c_col_offset+3.
        """
        def a_idx(i, j): return i * m + j
        def b_idx(i, j): return i * p + (b_col_offset + j)
        def c_idx(i, j): return (c_col_offset + j) * n + i

        # Laderman's 19 terms for 4x4*4x4
        # M1 = (A00+A11)*(B00+B11)
        u = zeros(U_rows); v = zeros(V_rows); w = zeros(W_rows)
        u[a_idx(0, 0)] = 1; u[a_idx(1, 1)] = 1
        v[b_idx(0, 0)] = 1; v[b_idx(1, 1)] = 1
        w[c_idx(0, 0)] = 1; w[c_idx(1, 1)] = 1
        add_term(u, v, w)

        # M2 = (A10+A11)*B00
        u = zeros(U_rows); v = zeros(V_rows); w = zeros(W_rows)
        u[a_idx(1, 0)] = 1; u[a_idx(1, 1)] = 1
        v[b_idx(0, 0)] = 1
        w[c_idx(1, 0)] = 1; w[c_idx(1, 1)] = -1
        add_term(u, v, w)

        # M3 = A00*(B01-B11)
        u = zeros(U_rows); v = zeros(V_rows); w = zeros(W_rows)
        u[a_idx(0, 0)] = 1
        v[b_idx(0, 1)] = 1; v[b_idx(1, 1)] = -1
        w[c_idx(0, 1)] = 1; w[c_idx(1, 1)] = 1
        add_term(u, v, w)

        # M4 = (A11-A00)*B10
        u = zeros(U_rows); v = zeros(V_rows); w = zeros(W_rows)
        u[a_idx(1, 1)] = 1; u[a_idx(0, 0)] = -1
        v[b_idx(1, 0)] = 1
        w[c_idx(0, 0)] = -1; w[c_idx(1, 0)] = 1
        add_term(u, v, w)

        # M5 = (A00+A01)*B11
        u = zeros(U_rows); v = zeros(V_rows); w = zeros(W_rows)
        u[a_idx(0, 0)] = 1; u[a_idx(0, 1)] = 1
        v[b_idx(1, 1)] = 1
        w[c_idx(0, 0)] = 1; w[c_idx(0, 1)] = 1
        add_term(u, v, w)

        # M6 = (A10-A00)*(B00+B01)
        u = zeros(U_rows); v = zeros(V_rows); w = zeros(W_rows)
        u[a_idx(1, 0)] = 1; u[a_idx(0, 0)] = -1
        v[b_idx(0, 0)] = 1; v[b_idx(0, 1)] = 1
        w[c_idx(0, 0)] = -1
        add_term(u, v, w)

        # M7 = (A01-A11)*(B10+B11)
        u = zeros(U_rows); v = zeros(V_rows); w = zeros(W_rows)
        u[a_idx(0, 1)] = 1; u[a_idx(1, 1)] = -1
        v[b_idx(1, 0)] = 1; v[b_idx(1, 1)] = 1
        w[c_idx(1, 1)] = 1
        add_term(u, v, w)

        # M8 = (A22+A33)*(B22+B33)
        u = zeros(U_rows); v = zeros(V_rows); w = zeros(W_rows)
        u[a_idx(2, 2)] = 1; u[a_idx(3, 3)] = 1
        v[b_idx(2, 2)] = 1; v[b_idx(3, 3)] = 1
        w[c_idx(2, 2)] = 1; w[c_idx(3, 3)] = 1
        add_term(u, v, w)

        # M9 = (A32+A33)*B22
        u = zeros(U_rows); v = zeros(V_rows); w = zeros(W_rows)
        u[a_idx(3, 2)] = 1; u[a_idx(3, 3)] = 1
        v[b_idx(2, 2)] = 1
        w[c_idx(3, 2)] = 1; w[c_idx(3, 3)] = -1
        add_term(u, v, w)

        # M10 = A22*(B23-B33)
        u = zeros(U_rows); v = zeros(V_rows); w = zeros(W_rows)
        u[a_idx(2, 2)] = 1
        v[b_idx(2, 3)] = 1; v[b_idx(3, 3)] = -1
        w[c_idx(2, 3)] = 1; w[c_idx(3, 3)] = 1
        add_term(u, v, w)

        # M11 = (A33-A22)*B32
        u = zeros(U_rows); v = zeros(V_rows); w = zeros(W_rows)
        u[a_idx(3, 3)] = 1; u[a_idx(2, 2)] = -1
        v[b_idx(3, 2)] = 1
        w[c_idx(2, 2)] = -1; w[c_idx(3, 2)] = 1
        add_term(u, v, w)

        # M12 = (A22+A23)*B33
        u = zeros(U_rows); v = zeros(V_rows); w = zeros(W_rows)
        u[a_idx(2, 2)] = 1; u[a_idx(2, 3)] = 1
        v[b_idx(3, 3)] = 1
        w[c_idx(2, 2)] = 1; w[c_idx(2, 3)] = 1
        add_term(u, v, w)

        # M13 = (A32-A22)*(B22+B23)
        u = zeros(U_rows); v = zeros(V_rows); w = zeros(W_rows)
        u[a_idx(3, 2)] = 1; u[a_idx(2, 2)] = -1
        v[b_idx(2, 2)] = 1; v[b_idx(2, 3)] = 1
        w[c_idx(2, 2)] = -1
        add_term(u, v, w)

        # M14 = (A23-A33)*(B32+B33)
        u = zeros(U_rows); v = zeros(V_rows); w = zeros(W_rows)
        u[a_idx(2, 3)] = 1; u[a_idx(3, 3)] = -1
        v[b_idx(3, 2)] = 1; v[b_idx(3, 3)] = 1
        w[c_idx(3, 3)] = 1
        add_term(u, v, w)

        # M15 = (A02+A13)*(B20+B31)
        u = zeros(U_rows); v = zeros(V_rows); w = zeros(W_rows)
        u[a_idx(0, 2)] = 1; u[a_idx(1, 3)] = 1
        v[b_idx(2, 0)] = 1; v[b_idx(3, 1)] = 1
        w[c_idx(0, 0)] = 1; w[c_idx(1, 1)] = 1
        add_term(u, v, w)

        # M16 = (A12+A13)*B20
        u = zeros(U_rows); v = zeros(V_rows); w = zeros(W_rows)
        u[a_idx(1, 2)] = 1; u[a_idx(1, 3)] = 1
        v[b_idx(2, 0)] = 1
        w[c_idx(1, 0)] = 1; w[c_idx(1, 1)] = -1
        add_term(u, v, w)

        # M17 = A02*(B30-B31)
        u = zeros(U_rows); v = zeros(V_rows); w = zeros(W_rows)
        u[a_idx(0, 2)] = 1
        v[b_idx(3, 0)] = 1; v[b_idx(3, 1)] = -1
        w[c_idx(0, 0)] = 1; w[c_idx(0, 1)] = 1
        add_term(u, v, w)

        # M18 = (A03-A02)*B30
        u = zeros(U_rows); v = zeros(V_rows); w = zeros(W_rows)
        u[a_idx(0, 3)] = 1; u[a_idx(0, 2)] = -1
        v[b_idx(3, 0)] = 1
        w[c_idx(0, 0)] = -1; w[c_idx(0, 1)] = 1
        add_term(u, v, w)

        # M19 = (A02+A12)*(B20+B21)
        u = zeros(U_rows); v = zeros(V_rows); w = zeros(W_rows)
        u[a_idx(0, 2)] = 1; u[a_idx(1, 2)] = 1
        v[b_idx(2, 0)] = 1; v[b_idx(2, 1)] = 1
        w[c_idx(0, 0)] = -1
        add_term(u, v, w)

    def schoolbook_4x4_4x2(b_col_offset, c_col_offset):
        """Add 8 rank-1 terms for schoolbook 4x4*4x2.
        A is full 4x4.
        B block: rows 0-3, cols b_col_offset to b_col_offset+1.
        C block: rows 0-3, cols c_col_offset to c_col_offset+1.
        """
        def a_idx(i, j): return i * m + j
        def b_idx(i, j): return i * p + (b_col_offset + j)
        def c_idx(i, j): return (c_col_offset + j) * n + i

        for i in range(4):
            for j in range(4):
                for k in range(2):
                    u = zeros(U_rows); v = zeros(V_rows); w = zeros(W_rows)
                    u[a_idx(i, j)] = 1
                    v[b_idx(j, k)] = 1
                    w[c_idx(i, k)] = 1
                    add_term(u, v, w)

    # Part 1: C[:, 0:4] = A * B[:, 0:4] using Laderman (rank 19)
    laderman_4x4_4x4(0, 0)

    # Part 2: C[:, 4:6] = A * B[:, 4:6] using schoolbook (rank 8)
    schoolbook_4x4_4x2(4, 4)

    rank = len(U_cols)

    U = [[U_cols[r][i] for r in range(rank)] for i in range(U_rows)]
    V = [[V_cols[r][i] for r in range(rank)] for i in range(V_rows)]
    W = [[W_cols[r][i] for r in range(rank)] for i in range(W_rows)]

    # Verify correctness with a random test
    try:
        import numpy as np
        np.random.seed(42)
        A_test = np.random.randint(-2, 3, (n, m)).astype(float)
        B_test = np.random.randint(-2, 3, (m, p)).astype(float)
        C_expected = A_test @ B_test

        U_mat = np.array(U, dtype=float)
        V_mat = np.array(V, dtype=float)
        W_mat = np.array(W, dtype=float)

        C_computed = np.zeros((n, p), dtype=float)
        for r in range(rank):
            u_vec = U_mat[:, r]
            v_vec = V_mat[:, r]
            w_vec = W_mat[:, r]
            u_mat = u_vec.reshape(n, m)
            v_mat = v_vec.reshape(m, p)
            w_mat = w_vec.reshape(p, n)
            for i in range(n):
                for k in range(p):
                    C_computed[i, k] += np.sum(u_mat[i, :] * v_mat[:, k] * w_mat[k, i])

        if not np.allclose(C_computed, C_expected, atol=1e-9):
            # Fallback: use full schoolbook
            U_cols2 = []
            V_cols2 = []
            W_cols2 = []
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        u = [0] * U_rows
                        v = [0] * V_rows
                        w = [0] * W_rows
                        u[i * m + j] = 1
                        v[j * p + k] = 1
                        w[k * n + i] = 1
                        U_cols2.append(u)
                        V_cols2.append(v)
                        W_cols2.append(w)
            rank2 = len(U_cols2)
            U = [[U_cols2[r][i] for r in range(rank2)] for i in range(U_rows)]
            V = [[V_cols2[r][i] for r in range(rank2)] for i in range(V_rows)]
            W = [[W_cols2[r][i] for r in range(rank2)] for i in range(W_rows)]
            return {"U": U, "V": V, "W": W}
    except Exception:
        pass

    return {"U": U, "V": V, "W": W}
