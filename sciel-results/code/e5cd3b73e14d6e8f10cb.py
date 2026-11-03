import numpy as np

DIMENSIONS = [4, 4, 6]

def construct():
    n, m, p = DIMENSIONS
    
    # Strassen 2x2x2 tensors
    # U: 4x7, V: 4x7, W: 4x7
    U_s = np.array([
        [1, 0, 1, 0, 1, -1, 0],
        [0, 0, 0, 0, 1, 0, 1],
        [0, 1, 0, 0, 0, 1, 0],
        [1, 1, 0, 1, 0, 0, -1]
    ])
    V_s = np.array([
        [1, 1, 0, -1, 0, 1, 0],
        [0, 0, 1, 0, 0, 1, 0],
        [0, 0, 0, 1, 0, 0, 1],
        [1, 0, -1, 0, 1, 0, 1]
    ])
    W_s = np.array([
        [1, 0, 0, 1, -1, 0, 1],
        [0, 0, 1, 0, 1, 0, 0],
        [0, 1, 0, 1, 0, 0, 0],
        [1, -1, 1, 0, 0, 1, 0]
    ])
    
    # Part 1: 4x4x4 decomposition (Rank 49)
    # Kronecker product of two Strassen tensors
    # U_444 = kron(U_s, U_s) ? 
    # Index mapping: i = 2*i1 + i2.
    # U_444[i, c] where c = 7*c1 + c2.
    # U_444[2*i1+i2, 7*c1+c2] = U_s[i2, c2] * U_s[i1, c1]
    
    U_444 = np.zeros((16, 49))
    V_444 = np.zeros((16, 49))
    W_444 = np.zeros((16, 49))
    
    for i1 in range(2):
        for i2 in range(2):
            i = 2*i1 + i2
            for c1 in range(7):
                for c2 in range(7):
                    c = 7*c1 + c2
                    U_444[i, c] = U_s[i1, c1] * U_s[i2, c2]
                    V_444[i, c] = V_s[i1, c1] * V_s[i2, c2]
                    W_444[i, c] = W_s[i1, c1] * W_s[i2, c2]
    
    # Part 2: 4x4x2 decomposition (Rank 28)
    # 4 independent 2x2x2 multiplications
    # A blocks: A11, A12, A21, A22
    # B blocks: B_top (rows 0,1), B_bot (rows 2,3)
    # C blocks: C_top (rows 0,1), C_bot (rows 2,3)
    # C_top = A11*B_top + A12*B_bot
    # C_bot = A21*B_top + A22*B_bot
    
    # We need to map these to the 4x4x2 tensor.
    # U_442: 16x28
    # V_442: 8x28
    # W_442: 8x28 (2 cols x 4 rows)
    
    U_442 = np.zeros((16, 28))
    V_442 = np.zeros((8, 28))
    W_442 = np.zeros((8, 28))
    
    # Term 1: A11 * B_top
    # A11 is rows 0,1; cols 0,1.
    # B_top is rows 0,1; cols 0,1 (of the 4x2 B, so cols 0,1).
    # C_top is rows 0,1; cols 0,1.
    
    # Term 2: A12 * B_bot
    # A12 is rows 0,1; cols 2,3.
    # B_bot is rows 2,3; cols 0,1.
    # C_top is rows 0,1; cols 0,1.
    
    # Term 3: A21 * B_top
    # A21 is rows 2,3; cols 0,1.
    # B_top is rows 0,1; cols 0,1.
    # C_bot is rows 2,3; cols 0,1.
    
    # Term 4: A22 * B_bot
    # A22 is rows 2,3; cols 2,3.
    # B_bot is rows 2,3; cols 0,1.
    # C_bot is rows 2,3; cols 0,1.
    
    # We will fill U_442, V_442, W_442 for each term.
    # Each term uses 7 rank-1 tensors.
    
    # Helper to add a Strassen term
    def add_strassen_term(U_mat, V_mat, W_mat, offset, a_row_start, a_col_start, b_row_start, b_col_start, c_row_start, c_col_start):
        # a_row_start, a_col_start: top-left of A block
        # b_row_start, b_col_start: top-left of B block
        # c_row_start, c_col_start: top-left of C block
        
        for s in range(7):
            # U
            for i2 in range(2):
                for i1 in range(2):
                    i = a_row_start + 2*i1 + i2 # Wait, A is 4x4.
                    # A block is 2x2.
                    # i1 is block row (0 or 1), i2 is row in block.
                    # Global row = a_row_start + 2*i1 + i2 ?
                    # No, A is 4x4.
                    # A11: rows 0,1; cols 0,1.
                    # A12: rows 0,1; cols 2,3.
                    # A21: rows 2,3; cols 0,1.
                    # A22: rows 2,3; cols 2,3.
                    
                    # So for A11, a_row_start=0, a_col_start=0.
                    # Global row = a_row_start + i1 (block row) * 2 + i2?
                    # No, A11 is top-left.
                    # Row 0,1. Col 0,1.
                    # i1=0, i2=0 -> Row 0.
                    # i1=0, i2=1 -> Row 1.
                    # i1=1, i2=0 -> Row 2.
                    # i1=1, i2=1 -> Row 3.
                    
                    # So Global Row = a_row_start + 2*i1 + i2.
                    # Global Col = a_col_start + 2*j1 + j2.
                    
                    # But A is 4x4.
                    # For A11, a_row_start=0, a_col_start=0.
                    # For A12, a_row_start=0, a_col_start=2.
                    # For A21, a_row_start=2, a_col_start=0.
                    # For A22, a_row_start=2, a_col_start=2.
                    
                    # So Global Row = a_row_start + 2*i1 + i2.
                    # Global Col = a_col_start + 2*j1 + j2.
                    
                    # Wait, U matrix is for A.
                    # U shape is (n*m, rank).
                    # U[i*m+j, c].
                    # So index is i*m + j.
                    
                    # For A11:
                    # i1=0, i2=0 -> i=0, j=0 -> idx 0
                    # i1=0, i2=1 -> i=0, j=1 -> idx 1
                    # i1=1, i2=0 -> i=1, j=0 -> idx 4
                    # i1=1, i2=1 -> i=1, j=1 -> idx 5
                    
                    # This matches 2*i1 + i2 for row, 2*j1 + j2 for col.
                    
                    pass # Logic below

        # U
        for s in range(7):
            col = offset + s
            for i1 in range(2):
                for i2 in range(2):
                    row = a_row_start + 2*i1 + i2
                    col_idx = a_col_start + 2*i1 + i2 # No, col is j.
                    # U index is i*m + j.
                    # i = a_row_start + i1? No.
                    # A is 4x4.
                    # Block A11 is rows 0,1 and cols 0,1.
                    # i1 is 0 or 1. i2 is 0 or 1.
                    # Global i = a_row_start + i1? No.
                    # If a_row_start=0, i1=0 -> i=0. i1=1 -> i=1.
                    # So Global i = a_row_start + i1.
                    # Global j = a_col_start + j1.
                    # Wait, i1, i2 are the Strassen indices.
                    # Strassen U_s[i, c] where i is 0..3.
                    # i=0 -> (0,0), i=1 -> (0,1), i=2 -> (1,0), i=3 -> (1,1).
                    # So i = 2*i1 + i2.
                    
                    # Global i = a_row_start + i1?
                    # If A11, a_row_start=0.
                    # i1=0 -> i=0. i1=1 -> i=1.
                    # So Global i = a_row_start + i1.
                    # Global j = a_col_start + j1.
                    
                    # Wait, U_s index is 0..3.
                    # U_s[0] corresponds to A11.
                    # U_s[1] corresponds to A12.
                    # U_s[2] corresponds to A21.
                    # U_s[3] corresponds to A22.
                    
                    # So for a block at (a_row_start, a_col_start),
                    # The local Strassen index i_s maps to global i, j.
                    # i_s = 0 -> local (0,0) -> global (a_row_start, a_col_start)
                    # i_s = 1 -> local (0,1) -> global (a_row_start, a_col_start+1)
                    # i_s = 2 -> local (1,0) -> global (a_row_start+1, a_col_start)
                    # i_s = 3 -> local (1,1) -> global (a_row_start+1, a_col_start+1)
                    
                    # So global_i = a_row_start + i_s // 2
                    # global_j = a_col_start + i_s % 2
                    
                    # U index = global_i * m + global_j.
                    
                    pass

    # Let's just fill it manually for the 4 terms.
    
    # Term 1: A11 * B_top
    # A11: rows 0,1; cols 0,1.
    # B_top: rows 0,1; cols 0,1.
    # C_top: rows 0,1; cols 0,1.
    
    for s in range(7):
        col = s
        # U
        for i_s in range(4):
            g_i = 0 + i_s // 2
            g_j = 0 + i_s % 2
            U_442[g_i * 4 + g_j, col] = U_s[i_s, s]
            
        # V
        for j_s in range(4):
            g_i = 0 + j_s // 2
            g_j = 0 + j_s % 2
            V_442[g_i * 2 + g_j, col] = V_s[j_s, s]
            
        # W
        for k_s in range(4):
            g_i = 0 + k_s // 2
            g_j = 0 + k_s % 2
            # W index is k*n + i.
            # k is col of C, i is row of C.
            # C is 4x2.
            # C_top is rows 0,1; cols 0,1.
            # So k is 0 or 1. i is 0 or 1.
            # W_442[k*n + i, col]
            W_442[g_j * 4 + g_i, col] = W_s[k_s, s]
            
    # Term 2: A12 * B_bot
    # A12: rows 0,1; cols 2,3.
    # B_bot: rows 2,3; cols 0,1.
    # C_top: rows 0,1; cols 0,1.
    
    for s in range(7):
        col = 7 + s
        # U
        for i_s in range(4):
            g_i = 0 + i_s // 2
            g_j = 2 + i_s % 2
            U_442[g_i * 4 + g_j, col] = U_s[i_s, s]
            
        # V
        for j_s in range(4):
            g_i = 2 + j_s // 2
            g_j = 0 + j_s % 2
            V_442[g_i * 2 + g_j, col] = V_s[j_s, s]
            
        # W
        for k_s in range(4):
            g_i = 0 + k_s // 2
            g_j = 0 + k_s % 2
            W_442[g_j * 4 + g_i, col] = W_s[k_s, s]
            
    # Term 3: A21 * B_top
    # A21: rows 2,3; cols 0,1.
    # B_top: rows 0,1; cols 0,1.
    # C_bot: rows 2,3; cols 0,1.
    
    for s in range(7):
        col = 14 + s
        # U
        for i_s in range(4):
            g_i = 2 + i_s // 2
            g_j = 0 + i_s % 2
            U_442[g_i * 4 + g_j, col] = U_s[i_s, s]
            
        # V
        for j_s in range(4):
            g_i = 0 + j_s // 2
            g_j = 0 + j_s % 2
            V_442[g_i * 2 + g_j, col] = V_s[j_s, s]
            
        # W
        for k_s in range(4):
            g_i = 2 + k_s // 2
            g_j = 0 + k_s % 2
            W_442[g_j * 4 + g_i, col] = W_s[k_s, s]
            
    # Term 4: A22 * B_bot
    # A22: rows 2,3; cols 2,3.
    # B_bot: rows 2,3; cols 0,1.
    # C_bot: rows 2,3; cols 0,1.
    
    for s in range(7):
        col = 21 + s
        # U
        for i_s in range(4):
            g_i = 2 + i_s // 2
            g_j = 2 + i_s % 2
            U_442[g_i * 4 + g_j, col] = U_s[i_s, s]
            
        # V
        for j_s in range(4):
            g_i = 2 + j_s // 2
            g_j = 0 + j_s % 2
            V_442[g_i * 2 + g_j, col] = V_s[j_s, s]
            
        # W
        for k_s in range(4):
            g_i = 2 + k_s // 2
            g_j = 0 + k_s % 2
            W_442[g_j * 4 + g_i, col] = W_s[k_s, s]

    # Combine
    # U: 16x49 and 16x28 -> 16x77
    # V: 16x49 and 8x28 -> 24x77
    # W: 16x49 and 8x28 -> 24x77
    
    U_final = np.hstack([U_444, U_442])
    
    # V_444 is 16x49.
    # V_442 is 8x28.
    # We need to pad V_444 to 24x49?
    # V_444 corresponds to B (4x4).
    # V_442 corresponds to B (4x2).
    # Total B is 4x6.
    # V shape is (m*p, rank) = (4*6, rank) = 24x77.
    
    # V_444 rows 0..15 correspond to B cols 0..3.
    # V_442 rows 0..7 correspond to B cols 4..5.
    
    V_final = np.zeros((24, 77))
    V_final[0:16, 0:49] = V_444
    V_final[0:8, 49:77] = V_442
    
    # W_444 is 16x49.
    # W_442 is 8x28.
    # W shape is (p*n, rank) = (6*4, rank) = 24x77.
    
    # W_444 rows 0..15 correspond to C cols 0..3.
    # W_442 rows 0..7 correspond to C cols 4..5.
    
    W_final = np.zeros((24, 77))
    W_final[0:16, 0:49] = W_444
    W_final[0:8, 49:77] = W_442
    
    return {
        "U": U_final.tolist(),
        "V": V_final.tolist(),
        "W": W_final.tolist()
    }

# EVOLVE-BLOCK-END
