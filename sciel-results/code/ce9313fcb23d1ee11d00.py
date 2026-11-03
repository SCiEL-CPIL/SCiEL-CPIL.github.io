import numpy as np
import time

def construct():
    start_time = time.time()
    deadline = start_time + 140

    n, m, p = 6, 6, 6
    bs = 2
    nb = n // bs
    rank = nb ** 3 * 7

    U = np.zeros((n * m, rank), dtype=int)
    V = np.zeros((m * p, rank), dtype=int)
    W = np.zeros((p * n, rank), dtype=int)

    # Strassen 2x2 coefficients: each term has (u_positions, v_positions, w_positions)
    # u_pos: list of (row_block_offset, col_block_offset)
    # v_pos: list of (row_block_offset, col_block_offset)
    # w_pos: list of (row_block_offset, col_block_offset)
    strassen_terms = [
        # term 0: (A11+A22)*(B11+B22)
        ([(0,0),(1,1)], [(0,0),(1,1)], [(0,0),(1,1)]),
        # term 1: (A21+A22)*B11
        ([(1,0),(1,1)], [(0,0)], [(1,0),(1,1)]),
        # term 2: A11*(B21-B11)
        ([(0,0)], [(0,1),(1,1)], [(0,1),(1,1)]),
        # term 3: A22*(B12-B22)
        ([(1,1)], [(1,0),(0,0)], [(0,0),(1,0)]),
        # term 4: (A11+A12)*B22
        ([(0,0),(0,1)], [(1,1)], [(0,0),(0,1)]),
        # term 5: (A21-A11)*(B11+B12)
        ([(1,0),(0,0)], [(0,0),(0,1)], [(1,1)]),
        # term 6: (A12-A22)*(B21+B22)
        ([(0,1),(1,1)], [(1,0),(1,1)], [(0,0)]),
    ]

    # Adjust term 2 and 3 signs
    # For term 2: A11*(B21-B11) => V has (0,1) with +1 and (1,1) with -1
    # For term 3: A22*(B12-B22) => V has (1,0) with +1 and (0,0) with -1
    # For term 5: (A21-A11) => U has (1,0) with +1 and (0,0) with -1
    # For term 6: (A12-A22) => U has (0,1) with +1 and (1,1) with -1
    # W signs:
    # term 0: C11 += T0, C12 -= T0, C21 -= T0, C22 += T0
    # term 1: C11 -= T1, C21 += T1
    # term 2: C12 += T2, C22 -= T2
    # term 3: C11 += T3, C21 -= T3  -- wait let me re-derive

    # Standard Strassen for C = A*B where A,B,C are 2x2:
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B12-B22)
    # M4 = A22(B21-B11)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    # C11 = M1 - M4 + M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6

    # Let me redefine with correct signs:
    strassen_terms_correct = [
        # (u_positions_with_signs, v_positions_with_signs, w_positions_with_signs)
        # M1
        ([(0,0,1),(1,1,1)], [(0,0,1),(1,1,1)], [(0,0,1),(1,1,1)]),
        # M2
        ([(1,0,1),(1,1,1)], [(0,0,1)], [(1,0,1),(1,1,-1)]),
        # M3
        ([(0,0,1)], [(0,1,1),(1,1,-1)], [(0,1,1),(1,1,1)]),
        # M4
        ([(1,1,1)], [(1,0,1),(0,0,-1)], [(0,0,1),(1,0,1)]),
        # M5
        ([(0,0,1),(0,1,1)], [(1,1,1)], [(0,0,-1),(0,1,1)]),
        # M6
        ([(1,0,1),(0,0,-1)], [(0,0,1),(0,1,1)], [(1,1,1)]),
        # M7
        ([(0,1,1),(1,1,-1)], [(1,0,1),(1,1,1)], [(0,0,1)]),
    ]

    col = 0
    for ib in range(nb):
        for jb in range(nb):
            for kb in range(nb):
                for term in strassen_terms_correct:
                    u_pos, v_pos, w_pos = term
                    for (r, c, sgn) in u_pos:
                        idx = (ib * bs + r) * m + (kb * bs + c)
                        U[idx, col] += sgn
                    for (r, c, sgn) in v_pos:
                        idx = (kb * bs + r) * p + (jb * bs + c)
                        V[idx, col] += sgn
                    for (r, c, sgn) in w_pos:
                        idx = (jb * bs + c) * n + (ib * bs + r)
                        W[idx, col] += sgn
                    col += 1

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
