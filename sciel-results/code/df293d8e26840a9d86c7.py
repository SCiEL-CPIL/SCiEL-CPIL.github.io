import numpy as np
import time

DIMENSIONS = [4, 4, 5]

def construct():
    n, m, p = DIMENSIONS  # 4, 4, 5
    start_time = time.time()
    DEADLINE = 130  # safety margin before 150s limit

    # Rank 65 decomposition:
    # - 49 terms for 4x4x4 via double-Strassen
    # - 16 terms for 4x4x1 via schoolbook
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

    # Inner Strassen 2x2x2: 7 multiplications
    # Each entry is (u_elems, v_elems, w_contribs)
    # u_elems: list of (row, col, coeff) for the 2x2 A block
    # v_elems: list of (row, col, coeff) for the 2x2 B block
    # w_contribs: list of (row, col, coeff) for the 2x2 C block
    inner_defs = [
        # M1 = (A00+A11)(B00+B11) -> C00 += M1, C11 += M1
        ([(0,0,1),(1,1,1)], [(0,0,1),(1,1,1)], [(0,0,1),(1,1,1)]),
        # M2 = (A10+A11)B00 -> C10 += M2, C11 -= M2
        ([(1,0,1),(1,1,1)], [(0,0,1)], [(1,0,1),(1,1,-1)]),
        # M3 = A00(B10-B00) -> C01 += M3, C11 += M3
        ([(0,0,1)], [(1,0,1),(0,0,-1)], [(0,1,1),(1,1,1)]),
        # M4 = A11(B01-B11) -> C00 += M4, C10 += M4
        ([(1,1,1)], [(0,1,1),(1,1,-1)], [(0,0,1),(1,0,1)]),
        # M5 = (A00+A01)B11 -> C00 -= M5, C01 += M5
        ([(0,0,1),(0,1,1)], [(1,1,1)], [(0,0,-1),(0,1,1)]),
        # M6 = (A10-A00)(B00+B01) -> C11 += M6
        ([(1,0,1),(0,0,-1)], [(0,0,1),(0,1,1)], [(1,1,1)]),
        # M7 = (A01-A11)(B10+B11) -> C00 += M7
        ([(0,1,1),(1,1,-1)], [(1,0,1),(1,1,1)], [(0,0,1)]),
    ]

    # Super-Strassen for 4x4x4: 7 outer multiplications
    # Each entry is (p_coeffs, q_coeffs, c_contribs)
    # p_coeffs: 4 coefficients for the 4 blocks of A (00,01,10,11)
    # q_coeffs: 4 coefficients for the 4 blocks of B (00,01,10,11)
    # c_contribs: list of (block_row, block_col, coeff) for C
    super_defs = [
        # M1 = (A00+A11)(B00+B11) -> C00 += M1, C11 += M1
        ([1,0,1,0], [1,0,1,0], [(0,0,1),(1,1,1)]),
        # M2 = (A10+A11)B00 -> C10 += M2, C11 -= M2
        ([0,0,1,1], [1,0,0,0], [(1,0,1),(1,1,-1)]),
        # M3 = A00(B10-B00) -> C01 += M3, C11 += M3
        ([1,0,0,0], [0,1,0,-1], [(0,1,1),(1,1,1)]),
        # M4 = A11(B01-B11) -> C00 += M4, C10 += M4
        ([0,0,0,1], [0,1,0,-1], [(0,0,1),(1,0,1)]),
        # M5 = (A00+A01)B11 -> C00 -= M5, C01 += M5
        ([1,1,0,0], [0,0,0,1], [(0,0,-1),(0,1,1)]),
        # M6 = (A10-A00)(B00+B01) -> C11 += M6
        ([-1,0,1,0], [1,1,0,0], [(1,1,1)]),
        # M7 = (A01-A11)(B10+B11) -> C00 += M7
        ([0,1,0,-1], [0,0,1,1], [(0,0,1)]),
    ]

    # Block index to global (row, col) mapping for 4x4 blocks
    # Block 0: rows 0-1, cols 0-1
    # Block 1: rows 0-1, cols 2-3
    # Block 2: rows 2-3, cols 0-1
    # Block 3: rows 2-3, cols 2-3
    def block_to_global(block_idx, local_r, local_c):
        if block_idx == 0:
            return local_r, local_c
        elif block_idx == 1:
            return local_r, 2 + local_c
        elif block_idx == 2:
            return 2 + local_r, local_c
        elif block_idx == 3:
            return 2 + local_r, 2 + local_c

    for s_idx, (p_bc, q_bc, c_contribs) in enumerate(super_defs):
        for i_idx, (p_elems, q_elems, r_contribs) in enumerate(inner_defs):
            u_list = []
            v_list = []
            w_list = []

            # Build U entries: combine super p_coeffs with inner p_elems
            # p_bc has 4 coefficients for blocks 0-3
            # p_elems has entries (local_r, local_c, inner_coeff)
            for b_idx, super_coeff in enumerate(p_bc):
                if super_coeff == 0:
                    continue
                for lr, lc, inner_coeff in p_elems:
                    gi, gj = block_to_global(b_idx, lr, lc)
                    u_list.append((gi, gj, super_coeff * inner_coeff))

            # Build V entries: combine super q_coeffs with inner q_elems
            for b_idx, super_coeff in enumerate(q_bc):
                if super_coeff == 0:
                    continue
                for lr, lc, inner_coeff in q_elems:
                    gj, gk = block_to_global(b_idx, lr, lc)
                    v_list.append((gj, gk, super_coeff * inner_coeff))

            # Build W entries: combine super c_contribs with inner r_contribs
            for br, bc, super_coeff in c_contribs:
                if super_coeff == 0:
                    continue
                for lr, lc, inner_coeff in r_contribs:
                    gi = br * 2 + lr
                    gk = bc * 2 + lc
                    w_list.append((gk, gi, super_coeff * inner_coeff))

            if u_list and v_list and w_list:
                add_term(col, u_list, v_list, w_list)
                col += 1

    # Part 2: 4x4x1 (rank 16) - schoolbook for the 5th column
    for i in range(4):
        for j in range(4):
            U[i * 4 + j][col] = 1
            V[j * 5 + 4][col] = 1
            W[4 * 4 + i][col] = 1
            col += 1

    if time.time() - start_time > DEADLINE - 5:
        pass

    return {"U": U, "V": V, "W": W}
