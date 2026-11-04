"""
Strategy: Direct block-Strassen composition for 4x4x7 matrix multiplication.

The 4x4x7 multiplication is split into:
- 4x4x4 part (output columns 0..3): decomposed via Strassen at the 2x2 block level.
  The 4x4x4 tensor = 7 "block multiplications" (Strassen M1..M7), each a 2x2x2 tensor.
  Each 2x2x2 tensor is decomposed into 7 rank-1 terms (Strassen 2x2x2).
  Total: 7 * 7 = 49 terms.
- 4x4x3 part (output columns 4..6): schoolbook, rank 48.

Total rank = 49 + 48 = 97 (vs schoolbook 112).
"""

import numpy as np
import time

DIMENSIONS = [4, 4, 7]

# EVOLVE-BLOCK-START

def construct():
    start_time = time.time()
    deadline = 130.0  # safety margin before 150s kill

    n, m, p = 4, 4, 7

    # Index helpers
    def a_idx(i, j):
        return i * m + j  # A is 4x4, row-major

    def b_idx(j, k):
        return j * p + k  # B is 4x7, row-major

    def c_idx(k, i):
        return k * n + i  # C is 4x7, trace convention W[k*n+i, r]

    # We build U (16 x R), V (28 x R), W (28 x R) as lists of (index, value) per rank column.
    # For efficiency, store as dict: {rank_col: {matrix_index: value}}
    U_dict = {}  # rank_col -> {a_index: value}
    V_dict = {}  # rank_col -> {b_index: value}
    W_dict = {}  # rank_col -> {c_index: value}

    rank_counter = 0

    def add_term(u_entries, v_entries, w_entries):
        """Add one rank-1 term. u/v/w are dicts {index: value}."""
        global rank_counter
        r = rank_counter
        rank_counter += 1
        U_dict[r] = u_entries
        V_dict[r] = v_entries
        W_dict[r] = w_entries

    # ============================================================
    # PART 1: 4x4x4 via block Strassen (49 terms)
    # ============================================================
    # A split into 2x2 blocks:
    #   A11 = rows 0-1, cols 0-1
    #   A12 = rows 0-1, cols 2-3
    #   A21 = rows 2-3, cols 0-1
    #   A22 = rows 2-3, cols 2-3
    # B split into 2x2 blocks (first 4 cols):
    #   B11 = rows 0-1, cols 0-1
    #   B12 = rows 0-1, cols 2-3
    #   B21 = rows 2-3, cols 0-1
    #   B22 = rows 2-3, cols 2-3
    # C (4x4 output, cols 0-3) split into 2x2 blocks:
    #   C11 = rows 0-1, cols 0-1
    #   C12 = rows 0-1, cols 2-3
    #   C21 = rows 2-3, cols 0-1
    #   C22 = rows 2-3, cols 2-3

    # Strassen 2x2x2: given 2x2 matrices X, Y, Z where Z = X*Y,
    # the 7 rank-1 terms use local indices (li, lj, lk) in {0,1}.
    # Local a index: li*2 + lj (within the 2x2 block)
    # Local b index: lj*2 + lk
    # Local c index: lk*2 + li

    # The 7 Strassen terms for 2x2x2 (local):
    # M1: (x00+x11)(y00+y11) -> z00+=1, z11+=1
    # M2: (x10+x11)y00       -> z10+=1
    # M3: x00(y01-y11)       -> z01+=1
    # M4: x11(y10-y00)       -> z00+=1, z11+=1
    # M5: (x00+x01)y11       -> z00+=1, z01+=1
    # M6: (x10-x00)(y00+y01) -> z11+=1
    # M7: (x01-x11)(y10+y11) -> z00+=1, z10+=1

    # For each Strassen block product (M1..M7), the X and Y are 2x2 blocks
    # (possibly sums of two blocks). We expand the sum linearly.
    # For a sum X = Xa + Xb, the U part of each rank-1 term gets contributions
    # from both Xa and Xb.

    # Define block index mappings:
    # Block (bi, bj) of A: global a indices for local (li, lj)
    #   a = (bi*2 + li) * 4 + (bj*2 + lj)
    # Block (bj, bk) of B: global b indices for local (lj, lk)
    #   b = (bj*2 + lj) * 7 + (bk*2 + lk)
    # Block (bk, bi) of C: global c indices for local (lk, li)
    #   c = (bk*2 + lk) * 4 + (bi*2 + li)

    def block_a(bi, bj, li, lj):
        return (bi * 2 + li) * 4 + (bj * 2 + lj)

    def block_b(bj, bk, lj, lk):
        return (bj * 2 + lj) * 7 + (bk * 2 + lk)

    def block_c(bk, bi, lk, li):
        return (bk * 2 + lk) * 4 + (bi * 2 + li)

    # The 7 Strassen block products for 4x4x4:
    # Each is: X * Y where X is a sum of A-blocks, Y is a sum of B-blocks.
    # X_blocks: list of (bi, bj, coeff)
    # Y_blocks: list of (bj, bk, coeff)
    # Output block: (bi, bk) for C

    strassen_products = [
        # M1: (A11+A22)(B11+B22) -> C11
        {'x': [(0,0,1),(1,1,1)], 'y': [(0,0,1),(1,1,1)], 'c': (0,0)},
        # M2: (A21+A22)B11 -> C21
        {'x': [(1,0,1),(1,1,1)], 'y': [(0,0,1)], 'c': (1,0)},
        # M3: A11(B12-B22) -> C12
        {'x': [(0,0,1)], 'y': [(0,1,1),(1,1,-1)], 'c': (0,1)},
        # M4: A22(B21-B11) -> C22
        {'x': [(1,1,1)], 'y': [(1,0,1),(0,0,-1)], 'c': (1,1)},
        # M5: (A11+A12)B22 -> C11
        {'x': [(0,0,1),(0,1,1)], 'y': [(1,1,1)], 'c': (0,0)},
        # M6: (A21-A11)(B11+B12) -> C22
        {'x': [(1,0,1),(0,0,-1)], 'y': [(0,0,1),(0,1,1)], 'c': (1,1)},
        # M7: (A12-A22)(B21+B22) -> C11
        {'x': [(0,1,1),(1,1,-1)], 'y': [(1,0,1),(1,1,1)], 'c': (0,0)},
    ]

    # For each block product, we need to decompose the 2x2x2 multiplication.
    # The X matrix is a linear combination of A-blocks, Y is a linear combination of B-blocks.
    # The 2x2x2 Strassen formula gives 7 rank-1 terms.
    # For each term, the U part involves specific local entries of X,
    # the V part involves specific local entries of Y,
    # and the W part involves specific local entries of C.
    # Since X = sum of blocks, each local entry of X is a sum of corresponding entries from each block.

    # The 7 Strassen terms (local structure):
    # Each term t has:
    #   u_pattern: list of (li, lj, coeff) - which local entries of X participate
    #   v_pattern: list of (lj, lk, coeff) - which local entries of Y participate
    #   w_pattern: list of (lk, li, coeff) - which local entries of C receive contribution

    strassen_2x2_terms = [
        # M1: (x00+x11)(y00+y11) -> z00, z11
        {'u': [(0,0,1),(1,1,1)], 'v': [(0,0,1),(1,1,1)], 'w': [(0,0,1),(1,1,1)]},
        # M2: (x10+x11)y00 -> z10
        {'u': [(1,0,1),(1,1,1)], 'v': [(0,0,1)], 'w': [(0,1,1)]},
        # M3: x00(y01-y11) -> z01
        {'u': [(0,0,1)], 'v': [(0,1,1),(1,1,-1)], 'w': [(1,0,1)]},
        # M4: x11(y10-y00) -> z00, z11
        {'u': [(1,1,1)], 'v': [(1,0,1),(0,0,-1)], 'w': [(0,0,1),(1,1,1)]},
        # M5: (x00+x01)y11 -> z00, z01
        {'u': [(0,0,1),(0,1,1)], 'v': [(1,1,1)], 'w': [(0,0,1),(1,0,1)]},
        # M6: (x10-x00)(y00+y01) -> z11
        {'u': [(1,0,1),(0,0,-1)], 'v': [(0,0,1),(0,1,1)], 'w': [(1,1,1)]},
        # M7: (x01-x11)(y10+y11) -> z00, z10
        {'u': [(0,1,1),(1,1,-1)], 'v': [(1,0,1),(1,1,1)], 'w': [(0,0,1),(0,1,1)]},
    ]

    # Now compose: for each of the 7 block products, for each of the 7 2x2 terms,
    # we get one rank-1 term in the global decomposition.
    # The U part: for each (li, lj, coeff_u) in u_pattern, for each (bi, bj, coeff_x) in x_blocks,
    #   global_a = block_a(bi, bj, li, lj), contribution = coeff_u * coeff_x
    # The V part: for each (lj, lk, coeff_v) in v_pattern, for each (bj, bk, coeff_y) in y_blocks,
    #   global_b = block_b(bj, bk, lj, lk), contribution = coeff_v * coeff_y
    # The W part: for each (lk, li, coeff_w) in w_pattern,
    #   global_c = block_c(bk_c, bi_c, lk, li) where (bi_c, bk_c) = c block,
    #   contribution = coeff_w

    for prod in strassen_products:
        x_blocks = prod['x']  # list of (bi, bj, coeff)
        y_blocks = prod['y']  # list of (bj, bk, coeff)
        bi_c, bk_c = prod['c']  # output C block

        for term in strassen_2x2_terms:
            u_entries = {}
            v_entries = {}
            w_entries = {}

            # U part
            for (li, lj, cu) in term['u']:
                for (bi, bj, cx) in x_blocks:
                    ga = block_a(bi, bj, li, lj)
                    val = cu * cx
                    u_entries[ga] = u_entries.get(ga, 0) + val

            # V part
            for (lj, lk, cv) in term['v']:
                for (bj, bk, cy) in y_blocks:
                    gb = block_b(bj, bk, lj, lk)
                    val = cv * cy
                    v_entries[gb] = v_entries.get(gb, 0) + val

            # W part
            for (lk, li, cw) in term['w']:
                gc = block_c(bk_c, bi_c, lk, li)
                w_entries[gc] = w_entries.get(gc, 0) + cw

            # Filter out zero entries
            u_entries = {k: v for k, v in u_entries.items() if v != 0}
            v_entries = {k: v for k, v in v_entries.items() if v != 0}
            w_entries = {k: v for k, v in w_entries.items() if v != 0}

            if u_entries and v_entries and w_entries:
                add_term(u_entries, v_entries, w_entries)

    # ============================================================
    # PART 2: 4x4x3 via schoolbook (48 terms)
    # ============================================================
    # Output columns 4, 5, 6 (k = 4, 5, 6)
    for i in range(4):
        for j in range(4):
            for k in range(4, 7):
                ga = a_idx(i, j)
                gb = b_idx(j, k)
                gc = c_idx(k, i)
                add_term({ga: 1}, {gb: 1}, {gc: 1})

    total_rank = rank_counter

    # Check time
    if time.time() > deadline:
        # Should not happen for this fixed construction, but safety
        pass

    # Convert to dense matrices
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]

    for r in range(total_rank):
        for idx, val in U_dict[r].items():
            U[idx][r] = val
        for idx, val in V_dict[r].items():
            V[idx][r] = val
        for idx, val in W_dict[r].items():
            W[idx][r] = val

    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
