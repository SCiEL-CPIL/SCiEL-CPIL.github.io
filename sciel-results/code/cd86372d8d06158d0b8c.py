import numpy as np

DIMENSIONS = [4, 4, 6]

def construct():
    n, m, p = DIMENSIONS
    # Rank 77: 49 for 4x4x4 part, 28 for 4x4x2 part
    rank = 77
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen steps for 2x2x2
    # (alpha, beta, gamma)
    # alpha: A block coeffs, beta: B block coeffs, gamma: C block coeffs
    # Indices (i, j) for 2x2 blocks
    strassen_steps_222 = [
        ({(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}),
        ({(1,0): 1, (1,1): 1}, {(0,0): 1}, {(1,0): 1, (1,1): -1}),
        ({(0,0): 1}, {(0,1): 1, (1,1): -1}, {(0,1): 1, (1,1): 1}),
        ({(1,1): 1}, {(1,0): 1, (0,0): -1}, {(0,0): 1, (1,0): 1}),
        ({(0,0): 1, (0,1): 1}, {(1,1): 1}, {(0,0): -1, (0,1): 1}),
        ({(1,0): 1, (0,0): -1}, {(0,0): 1, (0,1): 1}, {(1,1): 1}),
        ({(0,1): 1, (1,1): -1}, {(1,0): 1, (1,1): 1}, {(0,0): 1})
    ]

    # Part 1: 4x4x4 (cols 0-3 of B, cols 0-3 of C)
    # A is 4x4, B1 is 4x4, C1 is 4x4
    # Split into 2x2 blocks of size 2x2
    col_start_1 = 0
    
    for s in range(7):
        alpha_4, beta_4, gamma_4 = strassen_steps_222[s]
        for s2 in range(7):
            alpha_2, beta_2, gamma_2 = strassen_steps_222[s2]
            
            col = col_start_1 + s * 7 + s2
            
            # U (A)
            for (bi, bj), val_a4 in alpha_4.items():
                for (si, sj), val_a2 in alpha_2.items():
                    r = (bi * 2 + si) * m + (bj * 2 + sj)
                    U[r][col] += val_a4 * val_a2
                    
            # V (B1)
            for (bi, bj), val_b4 in beta_4.items():
                for (si, sj), val_b2 in beta_2.items():
                    r = (bi * 2 + si) * p + (bj * 2 + sj)
                    V[r][col] += val_b4 * val_b2
                    
            # W (C1)
            for (bi, bj), val_c4 in gamma_4.items():
                for (si, sj), val_c2 in gamma_2.items():
                    r = (bj * 2 + sj) * n + (bi * 2 + si)
                    W[r][col] += val_c4 * val_c2

    # Part 2: 4x4x2 (cols 4-5 of B, cols 4-5 of C)
    # A is 4x4, B2 is 4x2, C2 is 4x2
    # Split A into 2x2 blocks of size 2x2
    # Split B2 into 2 blocks of size 2x2 (top, bottom)
    # Split C2 into 2 blocks of size 2x2 (top, bottom)
    
    col_start_2 = 49
    
    # Multiplications:
    # C_top = A_00 B_top + A_01 B_bot
    # C_bot = A_10 B_top + A_11 B_bot
    
    # This requires 4 multiplications of 2x2x2
    # M1: A_00 * B_top
    # M2: A_01 * B_bot
    # M3: A_10 * B_top
    # M4: A_11 * B_bot
    
    mults = [
        # (A_block_idx, B_block_idx, C_block_idx)
        # A_block: (0,0) is top-left, (0,1) is top-right, (1,0) is bot-left, (1,1) is bot-right
        # B_block: 0 is top, 1 is bot
        # C_block: 0 is top, 1 is bot
        ((0,0), 0, 0), # A_00 * B_top -> C_top
        ((0,1), 1, 0), # A_01 * B_bot -> C_top
        ((1,0), 0, 1), # A_10 * B_top -> C_bot
        ((1,1), 1, 1)  # A_11 * B_bot -> C_bot
    ]
    
    for m_idx, (a_block, b_block, c_block) in enumerate(mults):
        for s2 in range(7):
            alpha_2, beta_2, gamma_2 = strassen_steps_222[s2]
            col = col_start_2 + m_idx * 7 + s2
            
            # U (A)
            for (si, sj), val_a2 in alpha_2.items():
                # A block is a_block (bi, bj)
                # Global row = bi*2 + si, col = bj*2 + sj
                r = (a_block[0] * 2 + si) * m + (a_block[1] * 2 + sj)
                U[r][col] += val_a2
                
            # V (B2)
            for (si, sj), val_b2 in beta_2.items():
                # B2 block is b_block (0 or 1)
                # If b_block=0, rows 0-1. If b_block=1, rows 2-3.
                # Cols are 0-1 relative to B2, which are 4-5 global.
                r = (b_block * 2 + si) * p + (4 + sj)
                V[r][col] += val_b2
                
            # W (C2)
            for (si, sj), val_c2 in gamma_2.items():
                # C2 block is c_block (0 or 1)
                # If c_block=0, rows 0-1. If c_block=1, rows 2-3.
                # Cols are 0-1 relative to C2, which are 4-5 global.
                # W index is col * n + row
                r = (4 + sj) * n + (c_block * 2 + si)
                W[r][col] += val_c2

    return {"U": U, "V": V, "W": W}
