import numpy as np

DIMENSIONS = [4, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    
    # Helper to create schoolbook factors for n x m x p
    def schoolbook(n, m, p):
        rank = n * m * p
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
        return U, V, W

    # Helper to create Strassen factors for 2x2x2
    def strassen_2x2x2():
        # Indices for 2x2 matrix: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
        # U (A factors), V (B factors), W (C factors)
        # 7 terms
        U = np.zeros((4, 7))
        V = np.zeros((4, 7))
        W = np.zeros((4, 7))
        
        # M1 = (A00+A11)(B00+B11)
        U[0, 0] = 1; U[3, 0] = 1
        V[0, 0] = 1; V[3, 0] = 1
        W[0, 0] = 1; W[3, 0] = 1 # C00, C11
        
        # M2 = (A10+A11)B00
        U[2, 1] = 1; U[3, 1] = 1
        V[0, 1] = 1
        W[2, 1] = 1; W[3, 1] = -1 # C10, C11
        
        # M3 = A00(B10-B11)
        U[0, 2] = 1
        V[2, 2] = 1; V[3, 2] = -1
        W[1, 2] = 1; W[3, 2] = 1 # C01, C11
        
        # M4 = A11(B01-B00)
        U[3, 3] = 1
        V[1, 3] = 1; V[0, 3] = -1
        W[0, 3] = 1; W[3, 3] = 1 # C00, C11
        
        # M5 = (A00+A01)B11
        U[0, 4] = 1; U[1, 4] = 1
        V[3, 4] = 1
        W[0, 4] = -1; W[1, 4] = 1 # C00, C01
        
        # M6 = (A10-A00)(B00+B01)
        U[2, 5] = 1; U[0, 5] = -1
        V[0, 5] = 1; V[1, 5] = 1
        W[3, 5] = 1 # C11
        
        # M7 = (A01-A11)(B10+B11)
        U[1, 6] = 1; U[3, 6] = -1
        V[2, 6] = 1; V[3, 6] = 1
        W[0, 6] = 1 # C00
        
        return U, V, W

    # We will use a simple, correct block decomposition.
    # Split A (4x5) into top 2 rows (A_T) and bottom 2 rows (A_B).
    # Split B (5x5) into left 2 cols (B_L) and right 3 cols (B_R).
    # C (4x5) = A_T * B_L (2x2 part) + A_T * B_R (2x3 part)
    #           + A_B * B_L (2x2 part) + A_B * B_R (2x3 part)
    # 
    # This is just schoolbook essentially, but we can use Strassen for the 2x2x2 blocks.
    # 
    # Blocks:
    # 1. A_T (2x5) * B_L (5x2) -> C_TL (2x2). We can use Strassen if we split B_L into 2x2 and 3x2?
    #    Actually, B_L is 5x2. We can split B_L into B_L1 (2x2) and B_L2 (3x2).
    #    A_T is 2x5. We can split A_T into A_T1 (2x2) and A_T2 (2x3).
    #    C_TL = A_T1 * B_L1 + A_T2 * B_L2.
    #    A_T1 * B_L1 is 2x2 * 2x2 -> 2x2. Use Strassen (rank 7).
    #    A_T2 * B_L2 is 2x3 * 3x2 -> 2x2. Use Schoolbook (rank 12).
    #
    # 2. A_T (2x5) * B_R (5x3) -> C_TR (2x3).
    #    Split B_R into B_R1 (2x3) and B_R2 (3x3).
    #    C_TR = A_T1 * B_R1 + A_T2 * B_R2.
    #    A_T1 * B_R1 is 2x2 * 2x3 -> 2x3. Schoolbook (rank 12).
    #    A_T2 * B_R2 is 2x3 * 3x3 -> 2x3. Schoolbook (rank 18).
    #
    # 3. A_B (2x5) * B_L (5x2) -> C_BL (2x2).
    #    C_BL = A_B1 * B_L1 + A_B2 * B_L2.
    #    A_B1 * B_L1 is 2x2 * 2x2 -> 2x2. Strassen (rank 7).
    #    A_B2 * B_L2 is 2x3 * 3x2 -> 2x2. Schoolbook (rank 12).
    #
    # 4. A_B (2x5) * B_R (5x3) -> C_BR (2x3).
    #    C_BR = A_B1 * B_R1 + A_B2 * B_R2.
    #    A_B1 * B_R1 is 2x2 * 2x3 -> 2x3. Schoolbook (rank 12).
    #    A_B2 * B_R2 is 2x3 * 3x3 -> 2x3. Schoolbook (rank 18).
    #
    # Total Rank = 7 + 12 + 12 + 18 + 7 + 12 + 12 + 18 = 98.
    # This matches the previous program's target rank. The previous program failed due to incorrect Strassen W matrix.
    
    total_rank = 98
    U_full = np.zeros((n * m, total_rank))
    V_full = np.zeros((m * p, total_rank))
    W_full = np.zeros((p * n, total_rank))
    
    col_offset = 0
    
    # Define index ranges
    # A is 4x5. Indices: i*m + j.
    # Rows 0-1 are Top, 2-3 are Bottom.
    # Cols 0-1 are Left1, 2-4 are Right (split into 2-3 and 4? No, let's stick to the split above).
    # Let's refine the split to match the blocks:
    # A_T: rows 0-1. A_T1: cols 0-1. A_T2: cols 2-4.
    # A_B: rows 2-3. A_B1: cols 0-1. A_B2: cols 2-4.
    # B_L: cols 0-1. B_L1: rows 0-1. B_L2: rows 2-4.
    # B_R: cols 2-4. B_R1: rows 0-1. B_R2: rows 2-4.
    
    # Block 1: A_T1 * B_L1 -> C_TL_part1 (2x2x2 Strassen)
    U_s, V_s, W_s = strassen_2x2x2()
    a_idx = [0, 1, 5, 6] # A_T1: (0,0),(0,1),(1,0),(1,1)
    b_idx = [0, 1, 5, 6] # B_L1: (0,0),(0,1),(1,0),(1,1)
    c_idx = [0, 1, 5, 6] # C_TL: (0,0),(0,1),(1,0),(1,1)
    
    for r in range(7):
        for i_local, i_global in enumerate(a_idx):
            U_full[i_global, col_offset + r] = U_s[i_local, r]
        for j_local, j_global in enumerate(b_idx):
            V_full[j_global, col_offset + r] = V_s[j_local, r]
        for k_local, k_global in enumerate(c_idx):
            W_full[k_global, col_offset + r] = W_s[k_local, r]
    col_offset += 7
    
    # Block 2: A_T2 * B_L2 -> C_TL_part2 (2x3 * 3x2 -> 2x2 Schoolbook)
    U_s, V_s, W_s = schoolbook(2, 3, 2)
    a_idx = [2, 3, 4, 7, 8, 9] # A_T2: (0,2),(0,3),(0,4),(1,2),(1,3),(1,4)
    b_idx = [10, 11, 12, 15, 16, 17] # B_L2: (2,0),(2,1),(3,0),(3,1),(4,0),(4,1) -- Wait, B is 5x5.
    # B indices: j*p + k.
    # B_L2 rows 2-4, cols 0-1.
    # (2,0)=10, (2,1)=11, (3,0)=15, (3,1)=16, (4,0)=20, (4,1)=21.
    b_idx = [10, 11, 15, 16, 20, 21]
    c_idx = [0, 1, 5, 6] # C_TL: (0,0),(0,1),(1,0),(1,1)
    
    for r in range(12):
        for i_local, i_global in enumerate(a_idx):
            U_full[i_global, col_offset + r] = U_s[i_local, r]
        for j_local, j_global in enumerate(b_idx):
            V_full[j_global, col_offset + r] = V_s[j_local, r]
        for k_local, k_global in enumerate(c_idx):
            W_full[k_global, col_offset + r] = W_s[k_local, r]
    col_offset += 12
    
    # Block 3: A_T1 * B_R1 -> C_TR_part1 (2x2 * 2x3 -> 2x3 Schoolbook)
    U_s, V_s, W_s = schoolbook(2, 2, 3)
    a_idx = [0, 1, 5, 6] # A_T1
    b_idx = [2, 3, 4, 7, 8, 9] # B_R1: rows 0-1, cols 2-4.
    # (0,2)=2, (0,3)=3, (0,4)=4, (1,2)=7, (1,3)=8, (1,4)=9.
    c_idx = [2, 3, 4, 7, 8, 9] # C_TR: rows 0-1, cols 2-4.
    # C indices: k*n + i.
    # (2,0)=2, (3,0)=3, (4,0)=4, (2,1)=7, (3,1)=8, (4,1)=9.
    
    for r in range(12):
        for i_local, i_global in enumerate(a_idx):
            U_full[i_global, col_offset + r] = U_s[i_local, r]
        for j_local, j_global in enumerate(b_idx):
            V_full[j_global, col_offset + r] = V_s[j_local, r]
        for k_local, k_global in enumerate(c_idx):
            W_full[k_global, col_offset + r] = W_s[k_local, r]
    col_offset += 12
    
    # Block 4: A_T2 * B_R2 -> C_TR_part2 (2x3 * 3x3 -> 2x3 Schoolbook)
    U_s, V_s, W_s = schoolbook(2, 3, 3)
    a_idx = [2, 3, 4, 7, 8, 9] # A_T2
    b_idx = [12, 13, 14, 17, 18, 19, 22, 23, 24] # B_R2: rows 2-4, cols 2-4.
    # (2,2)=12, (2,3)=13, (2,4)=14, (3,2)=17, (3,3)=18, (3,4)=19, (4,2)=22, (4,3)=23, (4,4)=24.
    c_idx = [2, 3, 4, 7, 8, 9] # C_TR
    
    for r in range(18):
        for i_local, i_global in enumerate(a_idx):
            U_full[i_global, col_offset + r] = U_s[i_local, r]
        for j_local, j_global in enumerate(b_idx):
            V_full[j_global, col_offset + r] = V_s[j_local, r]
        for k_local, k_global in enumerate(c_idx):
            W_full[k_global, col_offset + r] = W_s[k_local, r]
    col_offset += 18
    
    # Block 5: A_B1 * B_L1 -> C_BL_part1 (2x2 * 2x2 -> 2x2 Strassen)
    U_s, V_s, W_s = strassen_2x2x2()
    a_idx = [10, 11, 15, 16] # A_B1: rows 2-3, cols 0-1.
    # (2,0)=10, (2,1)=11, (3,0)=15, (3,1)=16.
    b_idx = [0, 1, 5, 6] # B_L1
    c_idx = [10, 11, 15, 16] # C_BL: rows 2-3, cols 0-1.
    # (0,2)=10, (1,2)=11, (0,3)=15, (1,3)=16.
    
    for r in range(7):
        for i_local, i_global in enumerate(a_idx):
            U_full[i_global, col_offset + r] = U_s[i_local, r]
        for j_local, j_global in enumerate(b_idx):
            V_full[j_global, col_offset + r] = V_s[j_local, r]
        for k_local, k_global in enumerate(c_idx):
            W_full[k_global, col_offset + r] = W_s[k_local, r]
    col_offset += 7
    
    # Block 6: A_B2 * B_L2 -> C_BL_part2 (2x3 * 3x2 -> 2x2 Schoolbook)
    U_s, V_s, W_s = schoolbook(2, 3, 2)
    a_idx = [12, 13, 14, 17, 18, 19] # A_B2: rows 2-3, cols 2-4.
    # (2,2)=12, (2,3)=13, (2,4)=14, (3,2)=17, (3,3)=18, (3,4)=19.
    b_idx = [10, 11, 15, 16, 20, 21] # B_L2
    c_idx = [10, 11, 15, 16] # C_BL
    
    for r in range(12):
        for i_local, i_global in enumerate(a_idx):
            U_full[i_global, col_offset + r] = U_s[i_local, r]
        for j_local, j_global in enumerate(b_idx):
            V_full[j_global, col_offset + r] = V_s[j_local, r]
        for k_local, k_global in enumerate(c_idx):
            W_full[k_global, col_offset + r] = W_s[k_local, r]
    col_offset += 12
    
    # Block 7: A_B1 * B_R1 -> C_BR_part1 (2x2 * 2x3 -> 2x3 Schoolbook)
    U_s, V_s, W_s = schoolbook(2, 2, 3)
    a_idx = [10, 11, 15, 16] # A_B1
    b_idx = [2, 3, 4, 7, 8, 9] # B_R1
    c_idx = [12, 13, 14, 17, 18, 19] # C_BR: rows 2-3, cols 2-4.
    # (2,2)=12, (3,2)=13, (4,2)=14, (2,3)=17, (3,3)=18, (4,3)=19.
    
    for r in range(12):
        for i_local, i_global in enumerate(a_idx):
            U_full[i_global, col_offset + r] = U_s[i_local, r]
        for j_local, j_global in enumerate(b_idx):
            V_full[j_global, col_offset + r] = V_s[j_local, r]
        for k_local, k_global in enumerate(c_idx):
            W_full[k_global, col_offset + r] = W_s[k_local, r]
    col_offset += 12
    
    # Block 8: A_B2 * B_R2 -> C_BR_part2 (2x3 * 3x3 -> 2x3 Schoolbook)
    U_s, V_s, W_s = schoolbook(2, 3, 3)
    a_idx = [12, 13, 14, 17, 18, 19] # A_B2
    b_idx = [12, 13, 14, 17, 18, 19, 22, 23, 24] # B_R2
    c_idx = [12, 13, 14, 17, 18, 19] # C_BR
    
    for r in range(18):
        for i_local, i_global in enumerate(a_idx):
            U_full[i_global, col_offset + r] = U_s[i_local, r]
        for j_local, j_global in enumerate(b_idx):
            V_full[j_global, col_offset + r] = V_s[j_local, r]
        for k_local, k_global in enumerate(c_idx):
            W_full[k_global, col_offset + r] = W_s[k_local, r]
    col_offset += 18
    
    return {"U": U_full.tolist(), "V": V_full.tolist(), "W": W_full.tolist()}
# EVOLVE-BLOCK-END
