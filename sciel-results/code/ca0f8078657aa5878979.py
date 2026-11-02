"""Rank-42 exact tensor decomposition for 2x4 @ 4x6 via Strassen blocks.

Strategy: decompose into 6 independent 2x2x2 blocks and apply Strassen's
7-multiplication algorithm to each, giving 6*7 = 42 rank-1 terms.
"""
import time

_START = time.time()
_LIMIT = 110.0  # seconds, safety margin before 120s evaluator kill

def construct():
    # Precomputed rank-42 decomposition.
    # Each of the 6 blocks (3 column-pairs of the middle dimension x {left,right} halves of A)
    # uses Strassen on a 2x2x2 matrix multiply.

    n, m, p = 2, 4, 6
    rank_total = 42

    U = [[0] * rank_total for _ in range(n * m)]   # 8 x 42
    V = [[0] * rank_total for _ in range(m * p)]   # 24 x 42
    W = [[0] * rank_total for _ in range(p * n)]   # 12 x 42

    # Strassen matrices for 2x2x2:
    # U maps (a00,a01,a10,a11) -> 7 intermediates
    # V maps (b00,b01,b10,b11) -> 7 intermediates
    # W maps 7 products -> (c00,c01,c10,c11)
    strassen_U = [
        {0: 1, 3: 1},   # M0 = (a00+a11)(b00+b11)
        {2: 1, 3: 1},   # M1 = (a10+a11)b00
        {0: 1},         # M2 = a00(b01-b11)
        {3: 1},         # M3 = a11(b10-b00)
        {0: 1, 1: 1},   # M4 = (a00+a01)b11
        {0: -1, 2: 1},  # M5 = (a10-a00)(b00+b01)
        {1: 1, 3: -1},  # M6 = (a11-a01)(b10+b11)
    ]
    strassen_V = [
        {0: 1, 3: 1},
        {0: 1},
        {1: 1, 3: -1},
        {2: 1, 0: -1},
        {3: 1},
        {0: 1, 1: 1},
        {2: 1, 3: 1},
    ]
    strassen_W = [
        {0: 1, 3: -1},   # c00 = M0 + M1 - M2 - M3
        {2: 1},          # c01 = M2 + M4
        {0: -1, 1: 1},   # c10 = M1 + M3
        {0: 1, 2: -1},   # c11 = M0 - M4 + M5 + M6
    ]
    # Fix W to standard Strassen output combination:
    # c00 = M0 + M1 - M2 - M3
    # c01 = M2 + M4
    # c10 = M1 + M3
    # c11 = M0 - M4 + M5 + M6
    strassen_W = [
        {0: 1, 1: 1, 2: -1, 3: -1},
        {2: 1, 4: 1},
        {1: 1, 3: 1},
        {0: 1, 4: -1, 5: 1, 6: 1},
    ]

    col_offset = 0
    for j_block in range(3):
        for side in range(2):  # 0 = left half of A, 1 = right half of A
            if side == 0:
                a_row_base = 0   # rows 0,1 of A
                b_row_base = 0   # rows 0,1 of B (for this j_block)
            else:
                a_row_base = 2   # rows 2,3 of A
                b_row_base = 2   # rows 2,3 of B (for this j_block)

            for prod_idx in range(7):
                col = col_offset + prod_idx

                # Fill U: local indices 0..3 map to (row, col) in the 2x2 block of A
                # local 0 -> (0,0), 1 -> (0,1), 2 -> (1,0), 3 -> (1,1)
                # A is 2x4: row i in [0,1], col j in [a_row_base, a_row_base+1]
                # But wait: A is n x m = 2 x 4. So A[i, j] where i in [0,1], j in [0,3].
                # U has shape (n*m, rank) = (8, 42). Row index = i*m + j = i*4 + j.
                local_U_map = {
                    0: 0 * 4 + (a_row_base + 0),
                    1: 0 * 4 + (a_row_base + 1),
                    2: 1 * 4 + (a_row_base + 0),
                    3: 1 * 4 + (a_row_base + 1),
                }
                for local_idx, sign in strassen_U[prod_idx].items():
                    U[local_U_map[local_idx]][col] += sign

                # Fill V: B is m x p = 4 x 6. Row index = b_row*6 + col.
                # B[b_row, b_col] where b_row in [b_row_base, b_row_base+1],
                # b_col in [2*j_block, 2*j_block+1]
                local_V_map = {
                    0: (b_row_base + 0) * 6 + (2 * j_block + 0),
                    1: (b_row_base + 0) * 6 + (2 * j_block + 1),
                    2: (b_row_base + 1) * 6 + (2 * j_block + 0),
                    3: (b_row_base + 1) * 6 + (2 * j_block + 1),
                }
                for local_idx, sign in strassen_V[prod_idx].items():
                    V[local_V_map[local_idx]][col] += sign

                # Fill W: C is p x n = 6 x 2. Row index = c_row*2 + c_col.
                # C[c_row, c_col] where c_row in [2*j_block, 2*j_block+1],
                # c_col in [0,1]
                local_W_map = {
                    0: (2 * j_block + 0) * 2 + 0,
                    1: (2 * j_block + 1) * 2 + 0,
                    2: (2 * j_block + 0) * 2 + 1,
                    3: (2 * j_block + 1) * 2 + 1,
                }
                for local_idx, sign in strassen_W[local_idx if local_idx < len(strassen_W) else 0].items() if False else strassen_W[min(local_idx, 3)].items():
                    pass

            # Re-do W filling properly:
            # strassen_W has 4 entries (one per output local index 0..3)
            for out_local in range(4):
                # We need to distribute strassen_W[out_local] across the 7 products
                # This means for each product k, W[local_W_map[out_local]][col_offset+k] += strassen_W[out_local][k] if k in strassen_W[out_local]
                pass

    # The above W filling got tangled. Let me redo the whole thing cleanly.
    U = [[0] * rank_total for _ in range(n * m)]
    V = [[0] * rank_total for _ in range(m * p)]
    W = [[0] * rank_total for _ in range(p * n)]

    col_offset = 0
    for j_block in range(3):
        for side in range(2):
            if side == 0:
                a_base = 0
                b_base = 0
            else:
                a_base = 2
                b_base = 2

            for prod_idx in range(7):
                col = col_offset + prod_idx

                # U
                local_U_map = {
                    0: 0 * 4 + (a_base + 0),
                    1: 0 * 4 + (a_base + 1),
                    2: 1 * 4 + (a_base + 0),
                    3: 1 * 4 + (a_base + 1),
                }
                for local_idx, sign in strassen_U[prod_idx].items():
                    U[local_U_map[local_idx]][col] += sign

                # V
                local_V_map = {
                    0: (b_base + 0) * 6 + (2 * j_block + 0),
                    1: (b_base + 0) * 6 + (2 * j_block + 1),
                    2: (b_base + 1) * 6 + (2 * j_block + 0),
                    3: (b_base + 1) * 6 + (2 * j_block + 1),
                }
                for local_idx, sign in strassen_V[prod_idx].items():
                    V[local_V_map[local_idx]][col] += sign

                # W: for each output position, add contribution from this product
                # strassen_W[out_local] is a dict {prod_k: coeff}
                local_W_map = {
                    0: (2 * j_block + 0) * 2 + 0,
                    1: (2 * j_block + 1) * 2 + 0,
                    2: (2 * j_block + 0) * 2 + 1,
                    3: (2 * j_block + 1) * 2 + 1,
                }
                for out_local in range(4):
                    w_dict = strassen_W[out_local]
                    if prod_idx in w_dict:
                        W[local_W_map[out_local]][col] += w_dict[prod_idx]

            col_offset += 7

    if time.time() - _START > _LIMIT:
        pass  # Should never trigger; decomposition is precomputed

    return {"U": U, "V": V, "W": W}
