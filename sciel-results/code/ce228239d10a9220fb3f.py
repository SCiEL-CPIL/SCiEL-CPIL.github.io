import time

DIMENSIONS = [2, 2, 16]

# EVOLVE-BLOCK-START
def construct():
    t0 = time.time()
    deadline = t0 + 90

    n, m, p = DIMENSIONS  # 2, 2, 16
    num_blocks = p // 2   # 8 blocks of width 2
    rank = 7 * num_blocks # 56

    # Build factor matrices as dense lists
    U = [[0.0] * rank for _ in range(n * m)]   # 4 x 56
    V = [[0.0] * rank for _ in range(m * p)]   # 32 x 56
    W = [[0.0] * rank for _ in range(p * n)]   # 32 x 56

    # Strassen 2x2x2 decomposition: 7 rank-1 terms
    # Each term: (a_coeffs, b_coeffs, c_coeffs)
    # a_coeffs: list of (row, col, val) for the 2x2 A-factor
    # b_coeffs: list of (row, col, val) for the 2x2 B-factor
    # c_coeffs: list of (row, col, val) for the 2x2 C-factor
    #
    # Standard Strassen:
    # M1 = (a00+a11)(b00+b11)
    # M2 = (a10+a11)(b00)
    # M3 = (a00)(b01-b11)
    # M4 = (a11)(b10-b00)
    # M5 = (a00+a01)(b11)
    # M6 = (a10-a00)(b00+b01)
    # M7 = (a01-a11)(b10+b11)
    #
    # C00 = M1 - M4 + M5 + M7
    # C01 = M3 + M4
    # C10 = M2 + M5
    # C11 = M1 - M2 + M3 + M6

    terms = [
        # (U_entries, V_entries, W_entries)
        # M1
        ([(0,0,1.0),(1,1,1.0)],
         [(0,0,1.0),(1,1,1.0)],
         [(0,0,1.0),(1,1,1.0)]),
        # M2
        ([(1,0,1.0),(1,1,1.0)],
         [(0,0,1.0)],
         [(0,1,1.0),(1,1,-1.0)]),
        # M3
        ([(0,0,1.0)],
         [(0,1,1.0),(1,1,-1.0)],
         [(1,0,1.0),(1,1,1.0)]),
        # M4
        ([(1,1,1.0)],
         [(1,0,1.0),(0,0,-1.0)],
         [(0,0,1.0),(0,1,1.0)]),
        # M5
        ([(0,0,1.0),(0,1,1.0)],
         [(1,1,1.0)],
         [(0,0,-1.0),(1,0,1.0)]),
        # M6
        ([(1,0,1.0),(0,0,-1.0)],
         [(0,0,1.0),(0,1,1.0)],
         [(1,1,1.0)]),
        # M7
        ([(0,1,1.0),(1,1,-1.0)],
         [(1,0,1.0),(1,1,1.0)],
         [(0,0,1.0)]),
    ]

    for block_idx in range(num_blocks):
        if time.time() > deadline:
            break
        k_base = 2 * block_idx
        for term_idx in range(7):
            col = block_idx * 7 + term_idx
            u_entries, v_entries, w_entries = terms[term_idx]

            # Fill U: index is (i, j) -> row i*m + j
            for (i, j, val) in u_entries:
                U[i * m + j][col] = val

            # Fill V: index is (j, k) -> row j*p + k, where k = k_base + k_local
            for (j, k_local, val) in v_entries:
                k = k_base + k_local
                V[j * p + k][col] = val

            # Fill W: index is (k, i') -> row k*n + i', where k = k_base + k_local
            for (k_local, ip, val) in w_entries:
                k = k_base + k_local
                W[k * n + ip][col] = val

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
