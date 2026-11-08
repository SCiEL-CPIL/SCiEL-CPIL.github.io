import numpy as np

def construct():
    # Exact rank-98 decomposition for 4x4 * 4x8 matrix multiplication.
    # Built from two independent 4x4x4 Strassen rank-49 blocks.
    # U: (16, 98), V: (32, 98), W: (32, 98)
    # W index convention: W[k*n+i, r] where n=4, so W[k*4+i, r]

    # Strassen 2x2x2 rank-7 base
    # U_2, V_2, W_2 are 4x7 (indices 0..3 = row*2+col)
    U_2 = np.zeros((4, 7), dtype=int)
    V_2 = np.zeros((4, 7), dtype=int)
    W_2 = np.zeros((4, 7), dtype=int)

    # c=0: (a00+a11)*(b00+b11) -> c00+c11
    U_2[0, 0] = 1; U_2[3, 0] = 1
    V_2[0, 0] = 1; V_2[3, 0] = 1
    W_2[0, 0] = 1; W_2[3, 0] = 1

    # c=1: (a10+a11)*b00 -> c10-c11
    U_2[2, 1] = 1; U_2[3, 1] = 1
    V_2[0, 1] = 1
    W_2[2, 1] = 1; W_2[3, 1] = -1

    # c=2: a00*(b10-b11) -> c01+c11
    U_2[0, 2] = 1
    V_2[1, 2] = 1; V_2[3, 2] = -1
    W_2[1, 2] = 1; W_2[3, 2] = 1

    # c=3: a11*(b01-b00) -> c00+c10
    U_2[3, 3] = 1
    V_2[2, 3] = 1; V_2[0, 3] = -1
    W_2[0, 3] = 1; W_2[2, 3] = 1

    # c=4: (a00+a01)*b11 -> -c00+c01
    U_2[0, 4] = 1; U_2[1, 4] = 1
    V_2[3, 4] = 1
    W_2[0, 4] = -1; W_2[1, 4] = 1

    # c=5: (a10-a00)*(b00+b01) -> c11
    U_2[2, 5] = 1; U_2[0, 5] = -1
    V_2[0, 5] = 1; V_2[1, 5] = 1
    W_2[3, 5] = 1

    # c=6: (a01-a11)*(b10+b11) -> c00
    U_2[1, 6] = 1; U_2[3, 6] = -1
    V_2[2, 6] = 1; V_2[3, 6] = 1
    W_2[0, 6] = 1

    # Block Strassen specs for 4x4x4 (2x2 block structure)
    # Each spec: list of (coeff, block_idx) where block_idx = bi*2+bj
    A_specs = [
        [(1, 0), (1, 3)],    # P0: A00 + A11
        [(1, 2), (1, 3)],    # P1: A10 + A11
        [(1, 0)],            # P2: A00
        [(1, 3)],            # P3: A11
        [(1, 0), (1, 1)],    # P4: A00 + A01
        [(1, 2), (-1, 0)],   # P5: A10 - A00
        [(1, 1), (-1, 3)],   # P6: A01 - A11
    ]
    B_specs = [
        [(1, 0), (1, 3)],    # P0: B00 + B11
        [(1, 0)],            # P1: B00
        [(1, 1), (-1, 3)],   # P2: B01 - B11
        [(1, 2), (-1, 0)],   # P3: B10 - B00
        [(1, 3)],            # P4: B11
        [(1, 0), (1, 1)],    # P5: B00 + B01
        [(1, 2), (1, 3)],    # P6: B10 + B11
    ]
    C_specs = [
        [(1, 0), (1, 3)],    # P0: C00 + C11
        [(1, 2), (-1, 3)],   # P1: C10 - C11
        [(1, 1), (1, 3)],    # P2: C01 + C11
        [(1, 0), (1, 2)],    # P3: C00 + C10
        [(-1, 0), (1, 1)],   # P4: -C00 + C01
        [(1, 3)],            # P5: C11
        [(1, 0)],            # P6: C00
    ]

    # Build 4x4x4 rank-49 decomposition
    U_44 = np.zeros((16, 49), dtype=int)
    V_44 = np.zeros((16, 49), dtype=int)
    W_44 = np.zeros((16, 49), dtype=int)

    for k_p in range(7):
        for m_s in range(7):
            col = k_p * 7 + m_s

            # U_44: A is 4x4, index = row*4+col
            for r in range(2):
                for c in range(2):
                    a_eff = r * 2 + c
                    cu = U_2[a_eff, m_s]
                    if cu == 0:
                        continue
                    for coeff_a, block_idx in A_specs[k_p]:
                        bi, bj = divmod(block_idx, 2)
                        gr = bi * 2 + r
                        gc = bj * 2 + c
                        U_44[gr * 4 + gc, col] += coeff_a * cu

            # V_44: B is 4x4, index = row*4+col
            for r in range(2):
                for c in range(2):
                    b_eff = r * 2 + c
                    cv = V_2[b_eff, m_s]
                    if cv == 0:
                        continue
                    for coeff_b, block_idx in B_specs[k_p]:
                        bi, bj = divmod(block_idx, 2)
                        gr = bi * 2 + r
                        gc = bj * 2 + c
                        V_44[gr * 4 + gc, col] += coeff_b * cv

            # W_44: C is 4x4, index = k*4+i (trace convention)
            for r in range(2):
                for c in range(2):
                    p_eff = r * 2 + c
                    cw = W_2[p_eff, m_s]
                    if cw == 0:
                        continue
                    for coeff_c, c_block_idx in C_specs[k_p]:
                        bi, bj = divmod(c_block_idx, 2)
                        gi = bi * 2 + r  # i index
                        gk = bj * 2 + c  # k index
                        W_44[gk * 4 + gi, col] += coeff_c * cw

    # Assemble 4x4 * 4x8 -> 4x8 decomposition (rank 98)
    # Two independent 4x4x4 blocks: left half (B cols 0-3) and right half (B cols 4-7)
    U = np.zeros((16, 98), dtype=int)
    V = np.zeros((32, 98), dtype=int)
    W = np.zeros((32, 98), dtype=int)

    # U: same A for both halves
    U[:, 0:49] = U_44
    U[:, 49:98] = U_44

    # V: B is 4x8, index = j*8+k
    # Left half: k=0..3 -> V_44 rows map to V[ j*8+k, 0:49 ]
    # Right half: k=4..7 -> V_44 rows map to V[ j*8+k, 49:98 ]
    for j in range(4):
        for k in range(4):
            v44 = j * 4 + k
            V[j * 8 + k, 0:49] = V_44[v44, 0:49]
            V[j * 8 + (k + 4), 49:98] = V_44[v44, 0:49]

    # W: C is 4x8, index = k*4+i (trace convention: W[k*n+i, r])
    # Left half: k=0..3 -> W_44 rows map to W[ k*4+i, 0:49 ]
    # Right half: k=4..7 -> W_44 rows map to W[ (k+4)*4+i, 49:98 ]
    for k in range(4):
        for i in range(4):
            w44 = k * 4 + i
            W[k * 4 + i, 0:49] = W_44[w44, 0:49]
            W[(k + 4) * 4 + i, 49:98] = W_44[w44, 0:49]

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
