import numpy as np

DIMENSIONS = [3, 3, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Rank 68 decomposition
    rank = 68
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Helper to add a Strassen 2x2x2 block
    # i_offset: global i offset (0)
    # j_offset: global j offset (0)
    # k_offset: global k offset
    def add_strassen_block(i_off, j_off, k_off):
        nonlocal col
        # Strassen coefficients
        # M1: A00+A11, B00+B11 -> C00, C11
        # M2: A10+A11, B00    -> C10, -C11
        # M3: A00, B01-B11    -> -C00, C01
        # M4: A11, B10-B00    -> C00, C10
        # M5: A00+A01, B11    -> -C00, C01
        # M6: A10-A00, B00+B01-> C11
        # M7: A01-A11, B10+B11-> C00
        
        strassen_terms = [
            # (u_coeffs, v_coeffs, w_coeffs)
            # u: (i,j) -> val, v: (j,k) -> val, w: (k,i) -> val
            # Indices are local 0,1
            ({(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}), # M1
            ({(1,0):1, (1,1):1}, {(0,0):1}, {(0,1):1, (1,1):-1}), # M2: C10 +=, C11 -=
            ({(0,0):1}, {(0,1):1, (1,1):-1}, {(0,0):-1, (1,0):1}), # M3: C00 -=, C01 +=
            ({(1,1):1}, {(1,0):1, (0,0):-1}, {(0,0):1, (0,1):1}), # M4: C00 +=, C10 +=
            ({(0,0):1, (0,1):1}, {(1,1):1}, {(0,0):-1, (1,0):1}), # M5: C00 -=, C01 +=
            ({(1,0):1, (0,0):-1}, {(0,0):1, (0,1):1}, {(1,1):1}), # M6: C11 +=
            ({(0,1):1, (1,1):-1}, {(1,0):1, (1,1):1}, {(0,0):1}), # M7: C00 +=
        ]
        
        for u_c, v_c, w_c in strassen_terms:
            # Fill U
            for (li, lj), val in u_c.items():
                gi = i_off + li
                gj = j_off + lj
                U[gi * m + gj][col] = val
            # Fill V
            for (lj, lk), val in v_c.items():
                gj = j_off + lj
                gk = k_off + lk
                V[gj * p + gk][col] = val
            # Fill W
            for (lk, li), val in w_c.items():
                gk = k_off + lk
                gi = i_off + li
                W[gk * n + gi][col] = val
            col += 1

    # 1. A_LL (2x2) * B_top (2x8)
    # Split B_top into 4 blocks of 2x2
    for k_block in range(4):
        add_strassen_block(0, 0, k_block * 2)
    
    # 2. A_LH (2x1) * B_bot (1x8)
    # A_LH indices: i in {0,1}, j=2
    # B_bot indices: j=2, k in {0..7}
    # C indices: i in {0,1}, k in {0..7}
    # Term: A[i,2] * B[2,k] -> C[i,k]
    for i in range(2):
        for k in range(8):
            U[i * m + 2][col] = 1
            V[2 * p + k][col] = 1
            W[k * n + i][col] = 1
            col += 1
            
    # 3. A_HL (1x2) * B_top (2x8)
    # A_HL indices: i=2, j in {0,1}
    # B_top indices: j in {0,1}, k in {0..7}
    # C indices: i=2, k in {0..7}
    # Term: A[2,j] * B[j,k] -> C[2,k]
    for j in range(2):
        for k in range(8):
            U[2 * m + j][col] = 1
            V[j * p + k][col] = 1
            W[k * n + 2][col] = 1
            col += 1

    # 4. A_HH (1x1) * B_bot (1x8)
    # A_HH indices: i=2, j=2
    # B_bot indices: j=2, k in {0..7}
    # C indices: i=2, k in {0..7}
    # Term: A[2,2] * B[2,k] -> C[2,k]
    for k in range(8):
        U[2 * m + 2][col] = 1
        V[2 * p + k][col] = 1
        W[k * n + 2][col] = 1
        col += 1

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
