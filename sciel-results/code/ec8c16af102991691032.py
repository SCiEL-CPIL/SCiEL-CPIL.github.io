import numpy as np
import time

def construct():
    n, m, p = 3, 4, 4
    rank = 44
    
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    # Local factors for 3x2x2 multiplication
    # A is 3x2, B is 2x2, C is 3x2
    # Local A indices: 0..5 (row * 2 + col)
    # Local B indices: 0..3 (row * 2 + col)
    # Local C indices: 0..5 (row * 2 + col)
    
    U_loc = np.zeros((6, 11))
    V_loc = np.zeros((4, 11))
    W_loc = np.zeros((6, 11))
    
    # Strassen part (cols 0-6)
    # M1
    U_loc[0, 0] = 1; U_loc[3, 0] = 1
    V_loc[0, 0] = 1; V_loc[3, 0] = 1
    W_loc[0, 0] = 1; W_loc[3, 0] = 1 # c00, c11
    
    # M2
    U_loc[2, 1] = 1; U_loc[3, 1] = 1
    V_loc[0, 1] = 1
    W_loc[2, 1] = 1; W_loc[3, 1] = -1 # c10, c11
    
    # M3
    U_loc[0, 2] = 1
    V_loc[1, 2] = 1; V_loc[3, 2] = -1
    W_loc[1, 2] = 1; W_loc[3, 2] = 1 # c01, c11
    
    # M4
    U_loc[3, 3] = 1
    V_loc[2, 3] = 1; V_loc[0, 3] = -1
    W_loc[0, 3] = 1; W_loc[2, 3] = 1 # c00, c10
    
    # M5
    U_loc[0, 4] = 1; U_loc[1, 4] = 1
    V_loc[3, 4] = 1
    W_loc[0, 4] = -1; W_loc[1, 4] = 1 # c00, c01
    
    # M6
    U_loc[2, 5] = 1; U_loc[0, 5] = -1
    V_loc[0, 5] = 1; V_loc[1, 5] = 1
    W_loc[3, 5] = 1 # c11
    
    # M7
    U_loc[1, 6] = 1; U_loc[3, 6] = -1
    V_loc[2, 6] = 1; V_loc[3, 6] = 1
    W_loc[0, 6] = 1 # c00
    
    # Extra part (cols 7-10) for Row 2 of A
    # M8 = a20 * b00 -> c20
    U_loc[4, 7] = 1
    V_loc[0, 7] = 1
    W_loc[4, 7] = 1
    
    # M9 = a21 * b10 -> c20
    U_loc[5, 8] = 1
    V_loc[2, 8] = 1
    W_loc[4, 8] = 1
    
    # M10 = a20 * b01 -> c21
    U_loc[4, 9] = 1
    V_loc[1, 9] = 1
    W_loc[5, 9] = 1
    
    # M11 = a21 * b11 -> c21
    U_loc[5, 10] = 1
    V_loc[3, 10] = 1
    W_loc[5, 10] = 1
    
    # Mapping function
    def add_block(block_idx, a_row_start, a_col_start, b_row_start, b_col_start, c_row_start, c_col_start):
        col_offset = block_idx * 11
        
        # Pre-compute global indices
        a_indices = [(a_row_start + r) * m + (a_col_start + c) for r in range(3) for c in range(2)]
        b_indices = [(b_row_start + r) * p + (b_col_start + c) for r in range(2) for c in range(2)]
        c_indices = [(c_col_start + c) * n + (c_row_start + r) for r in range(3) for c in range(2)]
        
        # Copy U_loc
        for i, idx in enumerate(a_indices):
            U[idx, col_offset:col_offset+11] += U_loc[i, :]
            
        # Copy V_loc
        for i, idx in enumerate(b_indices):
            V[idx, col_offset:col_offset+11] += V_loc[i, :]
            
        # Copy W_loc
        for i, idx in enumerate(c_indices):
            W[idx, col_offset:col_offset+11] += W_loc[i, :]

    # Block 0: A_L (cols 0,1) * B_11 (rows 0,1, cols 0,1) -> C_L (cols 0,1)
    add_block(0, 0, 0, 0, 0, 0, 0)
    
    # Block 1: A_R (cols 2,3) * B_21 (rows 2,3, cols 0,1) -> C_L (cols 0,1)
    add_block(1, 0, 2, 2, 0, 0, 0)
    
    # Block 2: A_L (cols 0,1) * B_12 (rows 0,1, cols 2,3) -> C_R (cols 2,3)
    add_block(2, 0, 0, 0, 2, 0, 2)
    
    # Block 3: A_R (cols 2,3) * B_22 (rows 2,3, cols 2,3) -> C_R (cols 2,3)
    add_block(3, 0, 2, 2, 2, 0, 2)
    
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
