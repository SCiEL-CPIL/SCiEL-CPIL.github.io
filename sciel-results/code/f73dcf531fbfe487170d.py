import numpy as np
import time

DIMENSIONS = [3, 4, 4]

def construct():
    start_time = time.time()
    deadline = start_time + 145.0
    
    n, m, p = DIMENSIONS
    
    # Rank 44 solution
    # 28 terms for C_top (2x4) using Strassen on 2x2 blocks
    # 16 terms for C_bottom (1x4) using Schoolbook
    
    rank = 44
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    # Helper to add Strassen terms for 2x2 multiplication
    # A_block_indices: list of 4 indices in A_flat (row-major)
    # B_block_indices: list of 4 indices in B_flat (row-major)
    # C_block_indices: list of 4 indices in C_flat (row-major)
    # start_col: starting column in U,V,W
    
    def add_strassen_2x2(A_idx, B_idx, C_idx, start_col):
        # A_idx: [A00, A01, A10, A11]
        # B_idx: [B00, B01, B10, B11]
        # C_idx: [C00, C01, C10, C11]
        
        # Strassen M1..M7
        # M1 = A00 * (B00 - B11)
        # M2 = (A00 + A01) * B11
        # M3 = (A10 - A00) * B00
        # M4 = A11 * (B10 - B00)
        # M5 = (A00 + A11) * (B00 + B11)
        # M6 = (A01 - A11) * (B10 + B11)
        # M7 = (A10 + A00) * (B00 - B10)
        
        # Reconstruction:
        # C00 = M1 + M4 - M6 + M7
        # C01 = M2 + M3
        # C10 = M5 + M6
        # C11 = M1 - M3 + M5 + M4
        
        # Term 1: M1
        U[A_idx[0], start_col] = 1
        V[B_idx[0], start_col] = 1
        V[B_idx[3], start_col] = -1
        W[C_idx[0], start_col] = 1
        W[C_idx[3], start_col] = 1
        
        # Term 2: M2
        U[A_idx[0], start_col+1] = 1
        U[A_idx[1], start_col+1] = 1
        V[B_idx[3], start_col+1] = 1
        W[C_idx[1], start_col+1] = 1
        
        # Term 3: M3
        U[A_idx[2], start_col+2] = 1
        U[A_idx[0], start_col+2] = -1
        V[B_idx[0], start_col+2] = 1
        W[C_idx[1], start_col+2] = 1
        W[C_idx[3], start_col+2] = -1
        
        # Term 4: M4
        U[A_idx[3], start_col+3] = 1
        V[B_idx[2], start_col+3] = 1
        V[B_idx[0], start_col+3] = -1
        W[C_idx[0], start_col+3] = 1
        W[C_idx[3], start_col+3] = 1
        
        # Term 5: M5
        U[A_idx[0], start_col+4] = 1
        U[A_idx[3], start_col+4] = 1
        V[B_idx[0], start_col+4] = 1
        V[B_idx[3], start_col+4] = 1
        W[C_idx[2], start_col+4] = 1
        W[C_idx[3], start_col+4] = 1
        
        # Term 6: M6
        U[A_idx[1], start_col+5] = 1
        U[A_idx[3], start_col+5] = -1
        V[B_idx[2], start_col+5] = 1
        V[B_idx[3], start_col+5] = 1
        W[C_idx[0], start_col+5] = -1
        W[C_idx[2], start_col+5] = 1
        
        # Term 7: M7
        U[A_idx[2], start_col+6] = 1
        U[A_idx[0], start_col+6] = 1
        V[B_idx[0], start_col+6] = 1
        V[B_idx[2], start_col+6] = -1
        W[C_idx[0], start_col+6] = 1
        
        return start_col + 7

    # Define block indices
    # A: 3x4
    # A_L: rows 0-1, cols 0-1 -> indices 0, 1, 4, 5
    A_L_idx = [0, 1, 4, 5]
    # A_R: rows 0-1, cols 2-3 -> indices 2, 3, 6, 7
    A_R_idx = [2, 3, 6, 7]
    # A_B: row 2, cols 0-3 -> indices 8, 9, 10, 11
    A_B_idx = [8, 9, 10, 11]
    
    # B: 4x4
    # B_TL: rows 0-1, cols 0-1 -> indices 0, 1, 4, 5
    B_TL_idx = [0, 1, 4, 5]
    # B_TR: rows 0-1, cols 2-3 -> indices 2, 3, 6, 7
    B_TR_idx = [2, 3, 6, 7]
    # B_BL: rows 2-3, cols 0-1 -> indices 8, 9, 12, 13
    B_BL_idx = [8, 9, 12, 13]
    # B_BR: rows 2-3, cols 2-3 -> indices 10, 11, 14, 15
    B_BR_idx = [10, 11, 14, 15]
    
    # C: 3x4
    # C_L: rows 0-1, cols 0-1 -> indices 0, 1, 4, 5
    C_L_idx = [0, 1, 4, 5]
    # C_R: rows 0-1, cols 2-3 -> indices 2, 3, 6, 7
    C_R_idx = [2, 3, 6, 7]
    # C_B: row 2, cols 0-3 -> indices 8, 9, 10, 11
    C_B_idx = [8, 9, 10, 11]
    
    col = 0
    
    # 1. C_L = A_L B_TL + A_R B_BL
    # Compute A_L B_TL (7 terms)
    col = add_strassen_2x2(A_L_idx, B_TL_idx, C_L_idx, col)
    # Compute A_R B_BL (7 terms)
    col = add_strassen_2x2(A_R_idx, B_BL_idx, C_L_idx, col)
    
    # 2. C_R = A_L B_TR + A_R B_BR
    # Compute A_L B_TR (7 terms)
    col = add_strassen_2x2(A_L_idx, B_TR_idx, C_R_idx, col)
    # Compute A_R B_BR (7 terms)
    col = add_strassen_2x2(A_R_idx, B_BR_idx, C_R_idx, col)
    
    # 3. C_B = A_B B (Schoolbook, 16 terms)
    # A_B is 1x4. B is 4x4. C_B is 1x4.
    # C_B[j] = sum_k A_B[k] B[k,j]
    # A_B[k] is at index 8+k.
    # B[k,j] is at index k*4 + j.
    # C_B[j] is at index 8+j.
    
    for j in range(4):
        for k in range(4):
            U[8+k, col] = 1
            V[k*4+j, col] = 1
            W[8+j, col] = 1
            col += 1
            
    return {"U": U, "V": V, "W": W}
