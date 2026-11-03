import time
import numpy as np

DIMENSIONS = [4, 4, 6]

def construct():
    start_time = time.time()
    deadline = start_time + 145.0

    n, m, p = DIMENSIONS  # 4, 4, 6

    # Base Strassen 2x2x2 rank-7 terms
    # u, v, w are dicts: {linear_index: coeff}
    # Indices for 2x2: 0, 1, 2, 3 (row-major: (r,c) -> r*2+c)
    def get_strassen_2x2_2():
        terms = []
        # M1 = (A00+A11)(B00+B11) -> C00+ , C11+
        terms.append(({0:1, 3:1}, {0:1, 3:1}, {0:1, 3:1, 2:-1})) 
        # M2 = (A10+A11)B00 -> C10+ , C11+
        terms.append(({2:1, 3:1}, {0:1}, {1:1, 3:1}))
        # M3 = A00(B01-B11) -> C00+ , C10+
        terms.append(({0:1}, {1:1, 3:-1}, {1:1}))
        # M4 = A11(B10-B00) -> C00- , C11-
        terms.append(({3:1}, {2:1, 0:-1}, {2:1}))
        # M5 = (A00+A01)B11 -> C00+ , C01+
        terms.append(({0:1, 1:1}, {3:1}, {0:1, 1:1}))
        # M6 = (A10-A00)(B00+B01) -> C10+ , C11+
        terms.append(({2:1, 0:-1}, {0:1, 1:1}, {2:1, 3:1}))
        # M7 = (A01-A11)(B10+B11) -> C00+ , C01+
        terms.append(({1:1, 3:-1}, {2:1, 3:1}, {0:1, 3:1}))
        return terms

    s2 = get_strassen_2x2_2()

    terms = []

    # --- Part 1: 4x4 x 4x4 x 4x4 (Rank 49) ---
    # Outer Strassen combinations for 2x2 block structure
    # Each combo: (list of (block_idx, sign) for A, list of (block_idx, sign) for B)
    # Block indices: 0=(0,0), 1=(0,1), 2=(1,0), 3=(1,1)
    outer_combos = [
        ([(0, 1), (3, 1)], [(0, 1), (3, 1)]),   # M1
        ([(2, 1), (3, 1)], [(0, 1)]),             # M2
        ([(0, 1)], [(1, 1), (3, -1)]),            # M3
        ([(3, 1)], [(2, 1), (0, -1)]),            # M4
        ([(0, 1), (1, 1)], [(3, 1)]),             # M5
        ([(2, 1), (0, -1)], [(0, 1), (1, 1)]),    # M6
        ([(1, 1), (3, -1)], [(2, 1), (3, 1)]),    # M7
    ]

    # Mapping from outer combo index to C blocks it contributes to
    # c_map[outer_idx] = list of (c_block_idx, coeff)
    # c_block_idx: 0=(0,0), 1=(0,1), 2=(1,0), 3=(1,1)
    c_map = {
        0: [(0, 1), (3, 1)],      # M1 -> C00+, C11+
        1: [(2, 1), (3, 1)],      # M2 -> C10+, C11+
        2: [(0, 1), (2, 1)],      # M3 -> C00+, C10+
        3: [(0, -1), (3, -1)],    # M4 -> C00-, C11-
        4: [(0, 1), (1, 1)],      # M5 -> C00+, C01+
        5: [(2, 1), (3, 1)],      # M6 -> C10+, C11+
        6: [(0, 1), (1, 1)]       # M7 -> C00+, C01+
    }

    for oi, (a_blocks, b_blocks) in enumerate(outer_combos):
        if time.time() > deadline: break
        for si in range(7):
            if time.time() > deadline: break
            u_inner, v_inner, w_inner = s2[si]

            u_coeff = {}
            for u_key, u_val in u_inner.items():
                ur, uc = u_key // 2, u_key % 2
                for blk, sign in a_blocks:
                    bi, bj = blk // 2, blk % 2
                    # Global A index: (2*bi+ur)*4 + (2*bj+uc)
                    idx = (2*bi + ur) * 4 + (2*bj + uc)
                    u_coeff[idx] = u_coeff.get(idx, 0) + sign * u_val

            v_coeff = {}
            for v_key, v_val in v_inner.items():
                vr, vc = v_key // 2, v_key % 2
                for blk, sign in b_blocks:
                    bi, bj = blk // 2, blk % 2
                    # Global B index (for first 4 cols): (2*bi+vr)*6 + (2*bj+vc)
                    idx = (2*bi + vr) * 6 + (2*bj + vc)
                    v_coeff[idx] = v_coeff.get(idx, 0) + sign * v_val

            w_coeff = {}
            for c_blk_idx, coeff in c_map[oi]:
                ci, cj = c_blk_idx // 2, c_blk_idx % 2
                for w_key, w_val in w_inner.items():
                    wr, wc = w_key // 2, w_key % 2
                    # Global C index: (2*cj+wc)*4 + (2*ci+wr)
                    # Note: W index convention is col*4 + row.
                    # C block (ci, cj) has rows 2*ci..2*ci+1, cols 2*cj..2*cj+1
                    idx = (2*cj + wc) * 4 + (2*ci + wr)
                    w_coeff[idx] = w_coeff.get(idx, 0) + coeff * w_val

            terms.append((u_coeff, v_coeff, w_coeff))

    # --- Part 2: 4x4 x 4x2 x 4x2 (Rank 28) ---
    # B_last2 is cols 4,5 of B.
    # Split A into 4 blocks. B_last2 into Top (rows 0-1) and Bot (rows 2-3).
    # C_last2 into Top (rows 0-1) and Bot (rows 2-3).
    
    # Products:
    # C_Top = A_00 * B_Top + A_01 * B_Bot
    # C_Bot = A_10 * B_Top + A_11 * B_Bot
    
    prod_defs = [
        # (a_block_idx, b_part_idx, c_block_idx)
        # a_block: 0=(0,0), 1=(0,1), 2=(1,0), 3=(1,1)
        # b_part: 0=Top, 1=Bot
        # c_block: 0=Top, 1=Bot
        (0, 0, 0), # A00 * BTop -> CTop
        (1, 1, 0), # A01 * BBot -> CTop
        (2, 0, 1), # A10 * BTop -> CBot
        (3, 1, 1), # A11 * BBot -> CBot
    ]

    for ai, bi_part, ci_part in prod_defs:
        if time.time() > deadline: break
        for si in range(7):
            if time.time() > deadline: break
            u_inner, v_inner, w_inner = s2[si]

            u_coeff = {}
            for u_key, u_val in u_inner.items():
                ur, uc = u_key // 2, u_key % 2
                ai_r, ai_c = ai // 2, ai % 2
                idx = (2*ai_r + ur) * 4 + (2*ai_c + uc)
                u_coeff[idx] = u_coeff.get(idx, 0) + u_val

            v_coeff = {}
            for v_key, v_val in v_inner.items():
                vr, vc = v_key // 2, v_key % 2
                # B rows: Top is 0-1, Bot is 2-3. Cols are 4,5.
                if bi_part == 0:
                    base_r = 0
                else:
                    base_r = 2
                idx = (base_r + vr) * 6 + (4 + vc)
                v_coeff[idx] = v_coeff.get(idx, 0) + v_val

            w_coeff = {}
            for w_key, w_val in w_inner.items():
                wr, wc = w_key // 2, w_key % 2
                # C rows: Top is 0-1, Bot is 2-3. Cols are 4,5.
                if ci_part == 0:
                    base_r = 0
                else:
                    base_r = 2
                # W index: col*4 + row
                idx = (4 + wc) * 4 + (base_r + wr)
                w_coeff[idx] = w_coeff.get(idx, 0) + w_val

            terms.append((u_coeff, v_coeff, w_coeff))

    rank = len(terms)

    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    for col, (u_c, v_c, w_c) in enumerate(terms):
        for idx, val in u_c.items():
            U[idx][col] = val
        for idx, val in v_c.items():
            V[idx][col] = val
        for idx, val in w_c.items():
            W[idx][col] = val

    return {"U": U, "V": V, "W": W}

