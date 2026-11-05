import numpy as np
import time

DIMENSIONS = [4, 5, 5]
_START = time.time()
_DEADLINE = 115.0


def construct():
    n, m, p = DIMENSIONS
    rank = 85
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    col = 0

    # Standard Strassen formulas for 2x2x2 (7 products)
    # Using (i,j) notation for 2x2 blocks
    S_A = [
        [(0, 0, 1), (1, 1, 1)],
        [(1, 0, 1), (1, 1, 1)],
        [(0, 0, 1)],
        [(1, 1, 1)],
        [(0, 0, 1), (0, 1, 1)],
        [(1, 0, 1), (0, 0, -1)],
        [(0, 1, 1), (1, 1, -1)],
    ]
    S_B = [
        [(0, 0, 1), (1, 1, 1)],
        [(0, 0, 1)],
        [(0, 1, 1), (1, 1, -1)],
        [(1, 0, 1), (0, 0, -1)],
        [(1, 1, 1)],
        [(0, 0, 1), (0, 1, 1)],
        [(1, 0, 1), (1, 1, 1)],
    ]
    S_C = [
        [(0, 0, 1), (1, 1, 1)],
        [(1, 0, 1), (1, 1, -1)],
        [(0, 1, 1), (1, 1, 1)],
        [(0, 0, 1), (1, 0, 1)],
        [(0, 0, -1), (0, 1, 1)],
        [(1, 1, 1)],
        [(0, 0, 1)],
    ]

    # 4x4x4 block structure: 2x2 blocks of 2x2
    block_row_col = [(0, 0), (0, 2), (2, 0), (2, 2)]

    def a_index(block, r, c):
        br, bc = block_row_col[block]
        return r + br, c + bc

    def b_index(block, r, c):
        br, bc = block_row_col[block]
        return r + br, c + bc

    def c_index(block, r, c):
        br, bc = block_row_col[block]
        return r + br, c + bc

    # Build 4x4x4 Strassen terms (outer 7 × inner 7 = 49)
    # Outer Strassen pattern for the 2x2 super-blocks
    outer_S_A = [
        {0: 1, 3: 1},
        {2: 1, 3: 1},
        {0: 1},
        {3: 1},
        {0: 1, 1: 1},
        {2: 1, 0: -1},
        {1: 1, 3: -1},
    ]
    outer_S_B = [
        {0: 1, 3: 1},
        {0: 1},
        {1: 1, 3: -1},
        {2: 1, 0: -1},
        {3: 1},
        {0: 1, 1: 1},
        {2: 1, 3: 1},
    ]
    outer_S_C = [
        {0: 1, 3: 1},
        {2: 1, 3: -1},
        {1: 1, 3: 1},
        {0: 1, 2: 1},
        {0: -1, 1: 1},
        {3: 1},
        {0: 1},
    ]

    # 49 terms for the 4x4x4 core
    for o in range(7):
        for s in range(7):
            u_vec = np.zeros(n * m)
            v_vec = np.zeros(m * p)
            w_vec = np.zeros(p * n)

            # Accumulate A contributions
            for blk_a, sa in outer_S_A[o].items():
                for (r, c, sr) in S_A[s]:
                    gr, gc = a_index(blk_a, r, c)
                    u_vec[gr * m + gc] += sa * sr

            # Accumulate B contributions
            for blk_b, sb in outer_S_B[o].items():
                for (r, c, sr) in S_B[s]:
                    gr, gc = b_index(blk_b, r, c)
                    v_vec[gr * p + gc] += sb * sr

            # Accumulate C contributions
            for blk_c, sc in outer_S_C[o].items():
                for (r, c, sr) in S_C[s]:
                    gr, gc = c_index(blk_c, r, c)
                    w_vec[gc * n + gr] += sc * sr

            U[:, col] = u_vec
            V[:, col] = v_vec
            W[:, col] = w_vec
            col += 1

    # 16 terms: A[i,k] * B[k,4] -> C[i,4], i,k in 0..3
    for i in range(4):
        for k in range(4):
            u_vec = np.zeros(n * m)
            v_vec = np.zeros(m * p)
            w_vec = np.zeros(p * n)
            u_vec[i * m + k] = 1
            v_vec[k * p + 4] = 1
            w_vec[4 * n + i] = 1
            U[:, col] = u_vec
            V[:, col] = v_vec
            W[:, col] = w_vec
            col += 1

    # 20 terms: A[i,4] * B[k,j] -> C[i,j], i in 0..3, j in 0..4
    for i in range(4):
        for j in range(5):
            u_vec = np.zeros(n * m)
            v_vec = np.zeros(m * p)
            w_vec = np.zeros(p * n)
            u_vec[i * m + 4] = 1
            v_vec[4 * p + j] = 1
            w_vec[j * n + i] = 1
            U[:, col] = u_vec
            V[:, col] = v_vec
            W[:, col] = w_vec
            col += 1

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
