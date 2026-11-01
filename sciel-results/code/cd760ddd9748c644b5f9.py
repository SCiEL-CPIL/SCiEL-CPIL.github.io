"""
Rank-35 exact decomposition for 2x2 * 2x10 matrix multiplication.

Approach:
- The 10 columns of B are grouped into 5 independent 2x2 blocks.
- Each 2x2 block is handled by a rank-7 Strassen decomposition.
- Total rank = 5 * 7 = 35.

Indexing:
- U has shape (n*m, rank) = (4, 35)
- V has shape (m*p, rank) = (20, 35)
- W has shape (p*n, rank) = (20, 35), with W[k*n + i, c]
"""

import time

DIMENSIONS = [2, 2, 10]


def construct():
    start_time = time.time()
    deadline = start_time + 120  # safety margin

    n, m, p = DIMENSIONS
    rank = 35

    U = [[0] * rank for _ in range(n * m)]   # 4 x 35
    V = [[0] * rank for _ in range(m * p)]   # 20 x 35
    W = [[0] * rank for _ in range(p * n)]   # 20 x 35

    # --- Strassen rank-7 terms for each 2x2 block ---
    # A = [[a, b], [c, d]], B_block = [[e, f], [g, h]]
    # C[0,0] = ae + bg, C[0,1] = af + bh, C[1,0] = ce + dg, C[1,1] = cf + dh
    #
    # M1 = (a+d)(e+h)
    # M2 = (c+d)e
    # M3 = a(f-h)
    # M4 = d(g-e)
    # M5 = (a+b)h
    # M6 = (c-a)(e+f)
    # M7 = (b-d)(g+h)
    #
    # C[0,0] = M1 + M4 - M5 + M7
    # C[0,1] = M3 + M5
    # C[1,0] = M2 + M4
    # C[1,1] = M1 - M2 + M3 + M6

    strassen_terms = [
        # (U_entries, V_entries, W_entries)
        # Term 0: M1 = (a+d)(e+h)
        (
            {(0, 0): 1, (1, 1): 1},
            {(0, 0): 1, (1, 1): 1},
            {(0, 0): 1, (1, 1): 1},
        ),
        # Term 1: M2 = (c+d)e
        (
            {(1, 0): 1, (1, 1): 1},
            {(0, 0): 1},
            {(0, 1): 1, (1, 1): -1},
        ),
        # Term 2: M3 = a(f-h)
        (
            {(0, 0): 1},
            {(0, 1): 1, (1, 1): -1},
            {(1, 0): 1, (1, 1): 1},
        ),
        # Term 3: M4 = d(g-e)
        (
            {(1, 1): 1},
            {(1, 0): 1, (0, 0): -1},
            {(0, 0): 1, (0, 1): 1},
        ),
        # Term 4: M5 = (a+b)h
        (
            {(0, 0): 1, (0, 1): 1},
            {(1, 1): 1},
            {(0, 0): -1, (1, 0): 1},
        ),
        # Term 5: M6 = (c-a)(e+f)
        (
            {(1, 0): 1, (0, 0): -1},
            {(0, 0): 1, (0, 1): 1},
            {(1, 1): 1},
        ),
        # Term 6: M7 = (b-d)(g+h)
        (
            {(0, 1): 1, (1, 1): -1},
            {(1, 0): 1, (1, 1): 1},
            {(0, 0): 1},
        ),
    ]

    num_blocks = p // 2  # 5 blocks

    for r in range(num_blocks):
        for t in range(7):
            col = r * 7 + t
            u_entries, v_entries, w_entries = strassen_terms[t]
            for (i, j), val in u_entries.items():
                U[i * m + j][col] = val
            for (j, k), val in v_entries.items():
                V[j * p + (2 * r + k)][col] = val
            for (k, i), val in w_entries.items():
                W[(2 * r + k) * n + i][col] = val

        # Check deadline after each block
        if time.time() > deadline - 2:
            break

    return {"U": U, "V": V, "W": W}
