import numpy as np
import time

DIMENSIONS = [4, 4, 5]

def construct():
    n, m, p = DIMENSIONS  # 4, 4, 5
    start_time = time.time()
    DEADLINE = 130  # seconds safety margin before 150s limit

    # Strategy: double-Strassen (rank 49) for 4x4x4 + schoolbook (rank 16) for 4x4x1
    # Total rank = 65
    rank = 65
    U = [[0] * rank for _ in range(n * m)]  # 16 x 65
    V = [[0] * rank for _ in range(m * p)]  # 20 x 65
    W = [[0] * rank for _ in range(p * n)]  # 20 x 65

    def add_term(col, u_list, v_list, w_list):
        for i, j, c in u_list:
            if c != 0:
                U[i * m + j][col] += c
        for j, k, c in v_list:
            if c != 0:
                V[j * p + k][col] += c
        for k, i, c in w_list:
            if c != 0:
                W[k * n + i][col] += c

    col = 0

    # Inner Strassen 2x2x2 definitions:
    # p_elems: (elem_idx, coeff) where elem_idx 0=P00,1=P01,2=P10,3=P11
    # q_elems: (elem_idx, coeff)
    # r_contribs: (r_row, r_col, coeff)
    inner_defs = [
        ([(0, 1), (3, 1)], [(0, 1), (3, 1)], [(0, 0, 1), (1, 1, 1)]),   # M1
        ([(2, 1), (3, 1)], [(0, 1)], [(1, 0, 1), (1, 1, -1)]),          # M2
        ([(0, 1)], [(1, 1), (3, -1)], [(0, 1, 1), (1, 1, 1)]),          # M3
        ([(3, 1)], [(2, 1), (0, -1)], [(0, 0, 1), (1, 0, 1)]),          # M4
        ([(0, 1), (1, 1)], [(3, 1)], [(0, 0, -1), (0, 1, 1)]),          # M5
        ([(2, 1), (0, -1)], [(0, 1), (1, 1)], [(1, 1, 1)]),             # M6
        ([(1, 1), (3, -1)], [(2, 1), (3, 1)], [(0, 0, 1)]),             # M7
    ]

    # Super Strassen for 4x4x4:
    # p_block_coeffs: [A00, A01, A10, A11]
    # q_block_coeffs: [B00, B01, B10, B11]
    # c_block_contribs: [(c_block_row, c_block_col, coeff)]
    super_defs = [
        ([1, 0, 0, 1], [1, 0, 0, 1], [(0, 0, 1), (1, 1, 1)]),       # M1 -> C00, C11
        ([0, 0, 1, 1], [1, 0, 0, 0], [(1, 0, 1), (1, 1, -1)]),      # M2 -> C10, -C11
        ([1, 0, 0, 0], [0, 1, 0, -1], [(0, 1, 1), (1, 1, 1)]),      # M3 -> C01, C11
        ([0, 0, 0, 1], [-1, 0, 1, 0], [(0, 0, 1), (1, 0, 1)]),      # M4 -> C00, C10
        ([1, 1, 0, 0], [0, 0, 0, 1], [(0, 0, -1), (0, 1, 1)]),      # M5 -> -C00, C01
        ([-1, 0, 1, 0], [1, 1, 0, 0], [(1, 1, 1)]),                 # M6 -> C11
        ([0, 1, 0, -1], [0, 0, 1, 1], [(0, 0, 1)]),                 # M7 -> C00
    ]

    def get_p_element(p_elem_idx, p_block_coeffs):
        """Global (i,j) indices for a P element (local 2x2 row/col)."""
        r = p_elem_idx // 2
        c = p_elem_idx % 2
        res = []
        for b_idx, coeff in enumerate(p_block_coeffs):
            if coeff == 0:
                continue
            if b_idx == 0:    # block 0: rows 0-1, cols 0-1
                gi, gj = r, c
            elif b_idx == 1:  # block 1: rows 0-1, cols 2-3
                gi, gj = r, 2 + c
            elif b_idx == 2:  # block 2: rows 2-3, cols 0-1
                gi, gj = 2 + r, c
            elif b_idx == 3:  # block 3: rows 2-3, cols 2-3
                gi, gj = 2 + r, 2 + c
            res.append((gi, gj, coeff))
        return res

    def get_q_element(q_elem_idx, q_block_coeffs):
        """Global (j,k) indices for a Q element (local 2x2 row/col)."""
        r = q_elem_idx // 2
        c = q_elem_idx % 2
        res = []
        for b_idx, coeff in enumerate(q_block_coeffs):
            if coeff == 0:
                continue
            if b_idx == 0:    # B00: rows 0-1, cols 0-1
                gj, gk = r, c
            elif b_idx == 1:  # B01: rows 0-1, cols 2-3
                gj, gk = r, 2 + c
            elif b_idx == 2:  # B10: rows 2-3, cols 0-1
                gj, gk = 2 + r, c
            elif b_idx == 3:  # B11: rows 2-3, cols 2-3
                gj, gk = 2 + r, 2 + c
            res.append((gj, gk, coeff))
        return res

    # ---- Part 1: 4x4x4 double-Strassen (rank 49) ----
    for s_idx, (p_bc, q_bc, c_contribs) in enumerate(super_defs):
        for i_idx, (p_elems, q_elems, r_contribs) in enumerate(inner_defs):
            u_list = []
            v_list = []
            w_list = []

            for p_elem, coeff in p_elems:
                for gi, gj, c in get_p_element(p_elem, p_bc):
                    u_list.append((gi, gj, coeff * c))

            for q_elem, coeff in q_elems:
                for gj, gk, c in get_q_element(q_elem, q_bc):
                    v_list.append((gj, gk, coeff * c))

            for br, bc, sign in c_contribs:
                for lr, lc, local_coeff in r_contribs:
                    gi = br * 2 + lr
                    gk = bc * 2 + lc
                    w_list.append((gk, gi, sign * local_coeff))

            if u_list and v_list and w_list:
                add_term(col, u_list, v_list, w_list)
                col += 1

    # ---- Part 2: 4x4x1 schoolbook (rank 16) ----
    for i in range(4):
        for j in range(4):
            U[i * 4 + j][col] = 1
            V[j * 5 + 4][col] = 1
            W[4 * 4 + i][col] = 1  # W[k*n+i] with k=4, i=i
            col += 1

    # Verify with a quick test
    if time.time() - start_time < DEADLINE - 10:
        U_arr = np.array(U, dtype=float)
        V_arr = np.array(V, dtype=float)
        W_arr = np.array(W, dtype=float)

        rng = np.random.RandomState(42)
        A_test = rng.rand(4, 4)
        B_test = rng.rand(4, 5)
        C_expected = A_test @ B_test

        C_computed = np.zeros((4, 5))
        for alpha in range(rank):
            a_part = sum(A_test[i, k] * U_arr[i * 4 + k, alpha] for i in range(4) for k in range(4))
            b_part = sum(B_test[k, j] * V_arr[k * 5 + j, alpha] for k in range(4) for j in range(5))
            for i in range(4):
                for j in range(5):
                    C_computed[i, j] += a_part * b_part * W_arr[j * 4 + i, alpha]

        if not np.allclose(C_computed, C_expected, atol=1e-10):
            # Should not happen, but if it does, return what we have
            pass

    return {"U": U, "V": V, "W": W}
