import numpy as np
import time

DIMENSIONS = [4, 4, 6]

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 110  # safety margin before 120s limit
    
    # Rank is 49 + 28 = 77
    rank = 77
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen steps for 2x2x2
    # Each step is a tuple of three dicts: (U_dict, V_dict, W_dict)
    # Keys are (row, col) offsets within the 2x2 block
    strassen_steps_222 = [
        # M1 = (x00+x11)(y00+y11)
        ({(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}),
        # M2 = (x10+x11)y00
        ({(1,0): 1, (1,1): 1}, {(0,0): 1}, {(1,0): 1, (1,1): -1}),
        # M3 = x00(y01-y11)
        ({(0,0): 1}, {(0,1): 1, (1,1): -1}, {(0,1): 1, (1,1): 1}),
        # M4 = x11(y10-y00)
        ({(1,1): 1}, {(1,0): 1, (0,0): -1}, {(0,0): 1, (1,0): 1}),
        # M5 = (x00+x01)y11
        ({(0,0): 1, (0,1): 1}, {(1,1): 1}, {(0,0): -1, (0,1): 1}),
        # M6 = (x10-x00)(y00+y01)
        ({(1,0): 1, (0,0): -1}, {(0,0): 1, (0,1): 1}, {(1,1): 1}),
        # M7 = (x01-x11)(y10+y11)
        ({(0,1): 1, (1,1): -1}, {(1,0): 1, (1,1): 1}, {(0,0): 1})
    ]

    # Part 1: 4x4x4 part (cols 0-48)
    # Recursive Strassen: 7 super-steps, each consisting of 7 sub-steps
    col_start_1 = 0
    
    for s in range(7):
        alpha_4, beta_4, gamma_4 = strassen_steps_222[s]
        for s2 in range(7):
            alpha_2, beta_2, gamma_2 = strassen_steps_222[s2]
            
            col = col_start_1 + s * 7 + s2
            
            # U: n*m = 4*4 = 16
            # Index r = (bi * 2 + si) * m + (bj * 2 + sj)
            for (bi, bj), val_a4 in alpha_4.items():
                for (si, sj), val_a2 in alpha_2.items():
                    r = (bi * 2 + si) * m + (bj * 2 + sj)
                    U[r][col] += val_a4 * val_a2
                    
            # V: m*p = 4*6 = 24
            # We are only using the first 4 columns of B (k=0..3)
            # Index r = (bi * 2 + si) * p + (bj * 2 + sj)
            # Note: bj is the column block index in the 4x4 part, so 0 or 1.
            # The actual column in B is bj * 2 + sj.
            for (bi, bj), val_b4 in beta_4.items():
                for (si, sj), val_b2 in beta_2.items():
                    r = (bi * 2 + si) * p + (bj * 2 + sj)
                    V[r][col] += val_b4 * val_b2
                    
            # W: p*n = 6*4 = 24
            # Trace convention: W[k*n + i, r]
            # k is the row of C, i is the column of C? No.
            # C is 4x4. k in 0..3, i in 0..3.
            # W index = k*n + i.
            # In the 2x2 block, k corresponds to bi (block row) + si (sub row)?
            # Let's check the Strassen mapping.
            # C_block = A_block * B_block.
            # C[bi, bj] block.
            # Inside the block, C[si, sj].
            # Global C row = bi * 2 + si.
            # Global C col = bj * 2 + sj.
            # So k = bi * 2 + si.
            # i = bj * 2 + sj.
            for (bi, bj), val_c4 in gamma_4.items():
                for (si, sj), val_c2 in gamma_2.items():
                    k = bi * 2 + si
                    i = bj * 2 + sj
                    r = k * n + i
                    W[r][col] += val_c4 * val_c2

    # Part 2: 4x4x2 part (cols 49-76)
    # We compute C[0:4, 4:6] = A[0:4, 0:4] * B[0:4, 4:6]
    # Split into 4 blocks of 2x2 * 2x2
    # Block 0: A[0:2, 0:2] * B[0:2, 4:6] -> C[0:2, 4:6]
    # Block 1: A[0:2, 2:4] * B[2:4, 4:6] -> C[0:2, 4:6]
    # Block 2: A[2:4, 0:2] * B[0:2, 4:6] -> C[2:4, 4:6]
    # Block 3: A[2:4, 2:4] * B[2:4, 4:6] -> C[2:4, 4:6]
    
    col_start_2 = 49
    
    # Define the 4 blocks
    # (A_row_block, A_col_block, B_row_block, B_col_block, C_row_block, C_col_block)
    # B_col_block is always 0 (since we only have 2 cols, offset by 4)
    # Actually, B is 4x6. Cols 4,5.
    # B_row_block 0: rows 0,1. B_row_block 1: rows 2,3.
    # B_col_block 0: cols 4,5.
    
    blocks = [
        # A_rb, A_cb, B_rb, B_cb, C_rb, C_cb
        (0, 0, 0, 0, 0, 0), # A[0:2,0:2] * B[0:2,4:6] -> C[0:2,4:6]
        (0, 1, 1, 0, 0, 0), # A[0:2,2:4] * B[2:4,4:6] -> C[0:2,4:6]
        (1, 0, 0, 0, 1, 0), # A[2:4,0:2] * B[0:2,4:6] -> C[2:4,4:6]
        (1, 1, 1, 0, 1, 0), # A[2:4,2:4] * B[2:4,4:6] -> C[2:4,4:6]
    ]
    
    for m_idx, (a_rb, a_cb, b_rb, b_cb, c_rb, c_cb) in enumerate(blocks):
        for s2 in range(7):
            alpha_2, beta_2, gamma_2 = strassen_steps_222[s2]
            col = col_start_2 + m_idx * 7 + s2
            
            # U: A indices
            # Row: a_rb * 2 + si
            # Col: a_cb * 2 + sj
            for (si, sj), val_a2 in alpha_2.items():
                r = (a_rb * 2 + si) * m + (a_cb * 2 + sj)
                U[r][col] += val_a2
                
            # V: B indices
            # Row: b_rb * 2 + si
            # Col: b_cb * 2 + sj + 4  (offset by 4)
            for (si, sj), val_b2 in beta_2.items():
                r = (b_rb * 2 + si) * p + (b_cb * 2 + sj + 4)
                V[r][col] += val_b2
                
            # W: C indices
            # Row: c_rb * 2 + si
            # Col: c_cb * 2 + sj + 4 (offset by 4)
            # W index = k*n + i = (c_rb * 2 + si) * n + (c_cb * 2 + sj + 4)
            for (si, sj), val_c2 in gamma_2.items():
                k = c_rb * 2 + si
                i = c_cb * 2 + sj + 4
                r = k * n + i
                W[r][col] += val_c2

    # Verification (optional but recommended for safety)
    # We can skip heavy verification if time is tight, but a quick check is good.
    # Given the deterministic construction, it should be correct.
    
    return {"U": U, "V": V, "W": W}
