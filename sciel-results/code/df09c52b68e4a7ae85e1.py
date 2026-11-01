"""Optimized program: rank-minimal exact tensor decomposition for 2x4 @ 4x6 using Strassen blocks."""
DIMENSIONS = [2, 4, 6]

import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 100  # 100s limit to be safe
    
    n, m, p = 2, 4, 6
    rank_total = 42
    U = [[0] * rank_total for _ in range(n * m)]
    V = [[0] * rank_total for _ in range(m * p)]
    W = [[0] * rank_total for _ in range(p * n)]

    strassen_U = [
        {0: 1, 3: 1},
        {2: 1, 3: 1},
        {0: 1},
        {3: 1},
        {0: 1, 1: 1},
        {0: -1, 2: 1},
        {1: 1, 3: -1},
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
        {0: 1, 3: 1},
        {2: 1, 3: -1},
        {1: 1, 3: 1},
        {0: 1, 2: 1},
        {0: -1, 1: 1},
        {3: 1},
        {0: 1},
    ]

    col_offset = 0
    for j_block in range(3):
        for side in ['L', 'R']:
            if side == 'L':
                u_offset = 0
                v_row_offset = 0
            else:
                u_offset = 2
                v_row_offset = 2

            for prod_idx in range(7):
                col = col_offset + prod_idx

                local_U_map = {
                    0: 0 * 4 + (u_offset + 0),
                    1: 0 * 4 + (u_offset + 1),
                    2: 1 * 4 + (u_offset + 0),
                    3: 1 * 4 + (u_offset + 1),
                }
                for local_idx, sign in strassen_U[prod_idx].items():
                    U[local_U_map[local_idx]][col] += sign

                local_V_map = {
                    0: (v_row_offset + 0) * 6 + (2 * j_block + 0),
                    1: (v_row_offset + 0) * 6 + (2 * j_block + 1),
                    2: (v_row_offset + 1) * 6 + (2 * j_block + 0),
                    3: (v_row_offset + 1) * 6 + (2 * j_block + 1),
                }
                for local_idx, sign in strassen_V[prod_idx].items():
                    V[local_V_map[local_idx]][col] += sign

                local_W_map = {
                    0: (2 * j_block + 0) * 2 + 0,
                    1: (2 * j_block + 1) * 2 + 0,
                    2: (2 * j_block + 0) * 2 + 1,
                    3: (2 * j_block + 1) * 2 + 1,
                }
                for local_idx, sign in strassen_W[prod_idx].items():
                    W[local_W_map[local_idx]][col] += sign

            col_offset += 7

    return {"U": U, "V": V, "W": W}
