import numpy as np

DIMENSIONS = [4, 4, 6]

def construct():
    n, m, p = DIMENSIONS
    # Rank is 49 + 28 = 77
    rank = 77
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # The 7 Strassen steps for 2x2x2 multiplication
    # Each step is (alpha, beta, gamma) where each is a dict mapping (i,j) to coefficient
    # A is 2x2, B is 2x2, C is 2x2
    # Standard Strassen:
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B21-B22)
    # M4 = A22(B12-B11)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    # C11 = M1 - M4 + M5 - M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    strassen_steps_222 = [
        # M1: (A11+A22)(B11+B22)
        ({(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}),
        # M2: (A21+A22)B11
        ({(1,0): 1, (1,1): 1}, {(0,0): 1}, {(1,0): 1, (1,1): -1}),
        # M3: A11(B21-B22)
        ({(0,0): 1}, {(0,1): 1, (1,1): -1}, {(0,1): 1, (1,1): 1}),
        # M4: A22(B12-B11)
        ({(1,1): 1}, {(1,0): 1, (0,0): -1}, {(0,0): 1, (1,0): 1}),
        # M5: (A11+A12)B22
        ({(0,0): 1, (0,1): 1}, {(1,1): 1}, {(0,0): -1, (0,1): 1}),
        # M6: (A21-A11)(B11+B12)
        ({(1,0): 1, (0,0): -1}, {(0,0): 1, (0,1): 1}, {(1,1): 1}),
        # M7: (A12-A22)(B21+B22)
        ({(0,1): 1, (1,1): -1}, {(1,0): 1, (1,1): 1}, {(0,0): 1})
    ]
    
    # Part 1: 4x4x4 part (cols 0-48)
    # A is 4x4, B1 is 4x4 (cols 0-3 of B), C1 is 4x4 (cols 0-3 of C)
    # Partition A into 2x2 blocks of size 2x2.
    # Partition B1 into 2x2 blocks of size 2x2.
    # Partition C1 into 2x2 blocks of size 2x2.
    
    col_start_1 = 0
    
    # For each of the 7 Strassen steps for the 4x4 level
    for s in range(7):
        alpha_4, beta_4, gamma_4 = strassen_steps_222[s]
        # For each of the 7 Strassen steps for the 2x2 level (inside each block multiplication)
        for s2 in range(7):
            alpha_2, beta_2, gamma_2 = strassen_steps_222[s2]
            
            col = col_start_1 + s * 7 + s2
            
            # Process U (A)
            for (bi, bj), val_a4 in alpha_4.items():
                # bi, bj are block indices in 4x4 A (0 or 1)
                # The block is 2x2.
                for (si, sj), val_a2 in alpha_2.items():
                    # Global row index in A
                    r = (bi * 2 + si) * m + (bj * 2 + sj)
                    U[r][col] += val_a4 * val_a2
                    
            # Process V (B1)
            for (bi, bj), val_b4 in beta_4.items():
                for (si, sj), val_b2 in beta_2.items():
                    # B1 is 4x4.
                    # Global row = bi*2 + si, global col = bj*2 + sj
                    # V index is row * p + col.
                    r = (bi * 2 + si) * p + (bj * 2 + sj)
                    V[r][col] += val_b4 * val_b2
                    
            # Process W (C1)
            for (bi, bj), val_c4 in gamma_4.items():
                for (si, sj), val_c2 in gamma_2.items():
                    # C1 is 4x4.
                    # Global row = bi*2 + si, global col = bj*2 + sj
                    # W index is col * n + row.
                    r = (bj * 2 + sj) * n + (bi * 2 + si)
                    W[r][col] += val_c4 * val_c2

    # Part 2: 4x4x2 part (cols 49-76)
    # A is 4x4, B2 is 4x2 (cols 4-5 of B), C2 is 4x2 (cols 4-5 of C)
    # We need 4 multiplications of 2x2x2:
    # M1: A00 * B0 -> C0 (top half)
    # M2: A01 * B1 -> C0 (top half)
    # M3: A10 * B0 -> C1 (bottom half)
    # M4: A11 * B1 -> C1 (bottom half)
    
    col_start_2 = 49
    
    mults = [
        ((0,0), 0, 0),  # A00, B0 (rows 0-1 of B2), C0 (rows 0-1 of C2)
        ((0,1), 1, 0),  # A01, B1 (rows 2-3 of B2), C0 (rows 0-1 of C2)
        ((1,0), 0, 1),  # A10, B0 (rows 0-1 of B2), C1 (rows 2-3 of C2)
        ((1,1), 1, 1)   # A11, B1 (rows 2-3 of B2), C1 (rows 2-3 of C2)
    ]
    
    for m_idx, (a_block, b_row_block, c_row_block) in enumerate(mults):
        for s2 in range(7):
            alpha_2, beta_2, gamma_2 = strassen_steps_222[s2]
            col = col_start_2 + m_idx * 7 + s2
            
            # Process U (A)
            for (si, sj), val_a2 in alpha_2.items():
                r = (a_block[0] * 2 + si) * m + (a_block[1] * 2 + sj)
                U[r][col] += val_a2
                
            # Process V (B2)
            for (si, sj), val_b2 in beta_2.items():
                # B2 global col = 4 + sj
                r = (b_row_block * 2 + si) * p + (4 + sj)
                V[r][col] += val_b2
                
            # Process W (C2)
            for (si, sj), val_c2 in gamma_2.items():
                # C2 global col = 4 + sj
                r = (4 + sj) * n + (c_row_block * 2 + si)
                W[r][col] += val_c2

    return {"U": U, "V": V, "W": W}
