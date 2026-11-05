import time
import sys

DIMENSIONS = [4, 4, 5]


def _build_double_strassen(U, V, W, col_start):
    """
    Build the 49 rank-1 terms for 4x4x4 matrix multiplication using double Strassen.
    A is 4x4, B is 4x4, C is 4x4.
    U: (i*4+j, col) for A[i,j]
    V: (j*5+k, col) for B[j,k] -- but for the 4x4x4 part, k only goes 0..3
    W: (k*4+i, col) for C[i,k] -- W[k*n+i] with n=4
    Returns the next column index.
    """
    col = col_start

    # Inner Strassen definitions for 2x2x2:
    # Each entry: (p_elems, q_elems, r_contribs)
    # p_elems: list of (elem_idx, coeff) -- elem_idx 0=P00,1=P01,2=P10,3=P11
    # q_elems: list of (elem_idx, coeff)
    # r_contribs: list of (r_row, r_col, coeff)
    inner_defs = [
        ([(0, 1), (3, 1)], [(0, 1), (3, 1)], [(0, 0, 1), (1, 1, 1)]),
        ([(2, 1), (3, 1)], [(0, 1)], [(1, 0, 1), (1, 1, -1)]),
        ([(0, 1)], [(1, 1), (3, -1)], [(0, 1, 1), (1, 1, 1)]),
        ([(3, 1)], [(2, 1), (0, -1)], [(0, 0, 1), (1, 0, 1)]),
        ([(0, 1), (1, 1)], [(3, 1)], [(0, 0, -1), (0, 1, 1)]),
        ([(2, 1), (0, -1)], [(0, 1), (1, 1)], [(1, 1, 1)]),
        ([(1, 1), (3, -1)], [(2, 1), (3, 1)], [(0, 0, 1)]),
    ]

    # Super Strassen definitions for 4x4x4:
    # Each entry: (p_block_coeffs, q_block_coeffs, c_block_contribs)
    # p_block_coeffs: [A00, A01, A10, A11] coefficients
    # q_block_coeffs: [B00, B01, B10, B11] coefficients
    # c_block_contribs: [(c_block_row, c_block_col, coeff)]
    super_defs = [
        ([1, 0, 0, 1], [1, 0, 0, 1], [(0, 0, 1), (1, 1, 1)]),
        ([0, 0, 1, 1], [1, 0, 0, 0], [(1, 0, 1), (1, 1, -1)]),
        ([1, 0, 0, 0], [0, 1, 0, -1], [(0, 1, 1), (1, 1, 1)]),
        ([0, 0, 0, 1], [-1, 0, 1, 0], [(0, 0, 1), (1, 0, 1)]),
        ([1, 1, 0, 0], [0, 0, 0, 1], [(0, 0, -1), (0, 1, 1)]),
        ([-1, 0, 1, 0], [1, 1, 0, 0], [(1, 1, 1)]),
        ([0, 1, 0, -1], [0, 0, 1, 1], [(0, 0, 1)]),
    ]

    def get_p_global(p_elem_idx, p_block_coeffs):
        """Map local 2x2 element index to global (i,j) pairs with coefficients."""
        r = p_elem_idx // 2
        c = p_elem_idx % 2
        res = []
        for b_idx, coeff in enumerate(p_block_coeffs):
            if coeff == 0:
                continue
            if b_idx == 0:    # A00: rows 0-1, cols 0-1
                gi, gj = r, c
            elif b_idx == 1:  # A01: rows 0-1, cols 2-3
                gi, gj = r, 2 + c
            elif b_idx == 2:  # A10: rows 2-3, cols 0-1
                gi, gj = 2 + r, c
            elif b_idx == 3:  # A11: rows 2-3, cols 2-3
                gi, gj = 2 + r, 2 + c
            res.append((gi, gj, coeff))
        return res

    def get_q_global(q_elem_idx, q_block_coeffs):
        """Map local 2x2 element index to global (j,k) pairs with coefficients."""
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

    for s_idx, (p_bc, q_bc, c_contribs) in enumerate(super_defs):
        for i_idx, (p_elems, q_elems, r_contribs) in enumerate(inner_defs):
            u_list = []
            v_list = []
            w_list = []

            for p_elem, coeff in p_elems:
                for gi, gj, c in get_p_global(p_elem, p_bc):
                    u_list.append((gi, gj, coeff * c))

            for q_elem, coeff in q_elems:
                for gj, gk, c in get_q_global(q_elem, q_bc):
                    v_list.append((gj, gk, coeff * c))

            for br, bc, sign in c_contribs:
                for lr, lc, local_coeff in r_contribs:
                    gi = br * 2 + lr
                    gk = bc * 2 + lc
                    w_list.append((gk, gi, sign * local_coeff))

            if u_list and v_list and w_list:
                for i, j, c in u_list:
                    if c != 0:
                        U[i * 4 + j][col] += c
                for j, k, c in v_list:
                    if c != 0:
                        V[j * 5 + k][col] += c
                for k, i, c in w_list:
                    if c != 0:
                        W[k * 4 + i][col] += c
                col += 1

    return col


def _build_schoolbook_4x4x1(U, V, W, col_start):
    """
    Build the 16 rank-1 terms for 4x4x1 (column 4 of B).
    C[i,4] = sum_j A[i,j] * B[j,4]
    W index: k*n+i = 4*4+i = 16+i
    Returns the next column index.
    """
    col = col_start
    for i in range(4):
        for j in range(4):
            U[i * 4 + j][col] = 1
            V[j * 5 + 4][col] = 1
            W[4 * 4 + i][col] = 1
            col += 1
    return col


def construct():
    start_time = time.time()
    DEADLINE = 110.0  # safety margin before 120s limit

    n, m, p = DIMENSIONS  # 4, 4, 5

    rank = 65
    U = [[0] * rank for _ in range(n * m)]   # 16 x 65
    V = [[0] * rank for _ in range(m * p)]   # 20 x 65
    W = [[0] * rank for _ in range(p * n)]   # 20 x 65

    # Part 1: 4x4x4 via double Strassen (49 terms)
    col = _build_double_strassen(U, V, W, 0)

    if time.time() - start_time > DEADLINE - 5:
        return {"U": U, "V": V, "W": W}

    # Part 2: 4x4x1 via schoolbook (16 terms)
    col = _build_schoolbook_4x4x1(U, V, W, col)

    return {"U": U, "V": V, "W": W}
