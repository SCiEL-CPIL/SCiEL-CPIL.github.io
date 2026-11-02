import time

DIMENSIONS = [2, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # n=2, m=4, p=4
    # Rank = 28
    rank = 28

    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    # Strassen coefficients for 2x2x2
    # M1 = (A00 + A11)(B00 + B11)
    # M2 = (A10 + A11) B00
    # M3 = A00 (B01 - B11)
    # M4 = A11 (B10 - B00)
    # M5 = (A00 + A01) B11
    # M6 = (A10 - A00)(B00 + B01)
    # M7 = (A01 - A11)(B10 + B11)

    # Output C:
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6

    strassen_steps = [
        # M1
        {
            'u': {0: 1, 3: 1},
            'v': {0: 1, 3: 1},
            'w': {0: 1, 3: 1}
        },
        # M2
        {
            'u': {2: 1, 3: 1},
            'v': {0: 1},
            'w': {2: 1, 3: -1}
        },
        # M3
        {
            'u': {0: 1},
            'v': {1: 1, 3: -1},
            'w': {1: 1, 3: 1}
        },
        # M4
        {
            'u': {3: 1},
            'v': {2: 1, 0: -1},
            'w': {0: 1, 2: 1}
        },
        # M5
        {
            'u': {0: 1, 1: 1},
            'v': {3: 1},
            'w': {0: -1, 1: 1}
        },
        # M6
        {
            'u': {2: 1, 0: -1},
            'v': {0: 1, 1: 1},
            'w': {3: 1}
        },
        # M7
        {
            'u': {1: 1, 3: -1},
            'v': {2: 1, 3: 1},
            'w': {0: 1}
        }
    ]

    def place_block(a_col_start, b_row_start, b_col_start, c_col_start, col_offset):
        for s in range(7):
            step = strassen_steps[s]
            col = col_offset + s

            for local_idx, coeff in step['u'].items():
                lr = local_idx // 2
                lc = local_idx % 2
                gi = lr
                gj = a_col_start + lc
                U[gi * m + gj][col] += coeff

            for local_idx, coeff in step['v'].items():
                lr = local_idx // 2
                lc = local_idx % 2
                gi = b_row_start + lr
                gj = b_col_start + lc
                V[gi * p + gj][col] += coeff

            for local_idx, coeff in step['w'].items():
                lr = local_idx // 2
                lc = local_idx % 2
                gk = c_col_start + lc
                gi = lr
                W[gk * n + gi][col] += coeff

    place_block(0, 0, 0, 0, 0)
    place_block(0, 0, 2, 2, 7)
    place_block(2, 2, 0, 0, 14)
    place_block(2, 2, 2, 2, 21)

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
