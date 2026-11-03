"""Rewritten program for 3x4 * 4x8 matrix multiplication with rank-88 decomposition."""
DIMENSIONS = [3, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Rank: 2 blocks of Strassen (2x2 * 2x8) + 2 blocks of schoolbook (1x2 * 2x8)
    # Strassen 2x2 * 2x8: 4 sub-blocks of 2x2*2x2, each rank 7 -> 28
    # Two such: 56
    # Schoolbook 1x2 * 2x8: 16 each
    # Two such: 32
    # Total: 88
    rank = 88

    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    col = 0

    def add_rank_unit(u_coeffs, v_coeffs, w_coeffs):
        nonlocal col
        if col >= rank:
            raise Exception("Rank overflow")
        for idx, val in u_coeffs:
            U[idx][col] += val
        for idx, val in v_coeffs:
            V[idx][col] += val
        for idx, val in w_coeffs:
            W[idx][col] += val
        col += 1

    def add_strassen_block(a_rows, a_cols, b_rows, b_cols, c_rows, c_cols):
        """
        Adds Strassen decomposition for a 2x2 A block times a 2x2 B block.
        """
        u_a = a_rows[0] * m + a_cols[0]
        u_b = a_rows[0] * m + a_cols[1]
        u_c = a_rows[1] * m + a_cols[0]
        u_d = a_rows[1] * m + a_cols[1]

        v_e = b_rows[0] * p + b_cols[0]
        v_f = b_rows[0] * p + b_cols[1]
        v_g = b_rows[1] * p + b_cols[0]
        v_h = b_rows[1] * p + b_cols[1]

        w_c11 = b_cols[0] * n + c_rows[0]
        w_c12 = b_cols[1] * n + c_rows[0]
        w_c21 = b_cols[0] * n + c_rows[1]
        w_c22 = b_cols[1] * n + c_rows[1]

        # M1 = (a+d)(e+h)
        add_rank_unit([(u_a, 1), (u_d, 1)], [(v_e, 1), (v_h, 1)], [(w_c11, 1), (w_c22, 1)])
        # M2 = (c+d)e
        add_rank_unit([(u_c, 1), (u_d, 1)], [(v_e, 1)], [(w_c21, 1), (w_c22, -1)])
        # M3 = a(f-h)
        add_rank_unit([(u_a, 1)], [(v_f, 1), (v_h, -1)], [(w_c12, 1), (w_c22, 1)])
        # M4 = d(g-e)
        add_rank_unit([(u_d, 1)], [(v_g, 1), (v_e, -1)], [(w_c11, 1), (w_c21, 1)])
        # M5 = (a+b)h
        add_rank_unit([(u_a, 1), (u_b, 1)], [(v_h, 1)], [(w_c11, -1), (w_c12, 1)])
        # M6 = (c-a)(e+f)
        add_rank_unit([(u_c, 1), (u_a, -1)], [(v_e, 1), (v_f, 1)], [(w_c22, 1)])
        # M7 = (b-d)(g+h)
        add_rank_unit([(u_b, 1), (u_d, -1)], [(v_g, 1), (v_h, 1)], [(w_c11, 1)])

    # Block 1: A[0:2, 0:2] * B[0:2, 0:8] -> C[0:2, 0:8]
    for k_start in range(0, 8, 2):
        add_strassen_block([0, 1], [0, 1], [0, 1], [k_start, k_start + 1], [0, 1], [k_start, k_start + 1])

    # Block 3: A[0:2, 2:4] * B[2:4, 0:8] -> C[0:2, 0:8]
    for k_start in range(0, 8, 2):
        add_strassen_block([0, 1], [2, 3], [2, 3], [k_start, k_start + 1], [0, 1], [k_start, k_start + 1])

    # Block 2: A[2:3, 0:2] * B[0:2, 0:8] -> C[2:3, 0:8]
    for k in range(8):
        add_rank_unit([(2 * m + 0, 1)], [(0 * p + k, 1)], [(k * n + 2, 1)])
        add_rank_unit([(2 * m + 1, 1)], [(1 * p + k, 1)], [(k * n + 2, 1)])

    # Block 4: A[2:3, 2:4] * B[2:4, 0:8] -> C[2:3, 0:8]
    for k in range(8):
        add_rank_unit([(2 * m + 2, 1)], [(2 * p + k, 1)], [(k * n + 2, 1)])
        add_rank_unit([(2 * m + 3, 1)], [(3 * p + k, 1)], [(k * n + 2, 1)])

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
