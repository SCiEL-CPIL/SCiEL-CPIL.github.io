import numpy as np
import time

def construct():
    start_time = time.time()
    deadline = 140.0  # Safety margin before 150s kill
    
    n, m, p = 3, 4, 4
    rank = 44
    
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    # Local factors for 3x2x2 multiplication
    # A is 3x2, B is 2x2, C is 3x2
    # Local A indices: r*2+c for r in 0..2, c in 0..1
    # Local B indices: r*2+c for r in 0..1, c in 0..1
    # Local C indices: c*3+r for c in 0..1, r in 0..2 (trace convention: W[k*n+i] where k=col, i=row)
    
    U_loc = np.zeros((6, 11))
    V_loc = np.zeros((4, 11))
    W_loc = np.zeros((6, 11))
    
    # Strassen part (cols 0-6) for 2x2x2
    # M1 = (a00+a10)*(b00+b10) -> c00 += M1, c11 += M1
    U_loc[0, 0] = 1; U_loc[2, 0] = 1
    V_loc[0, 0] = 1; V_loc[2, 0] = 1
    W_loc[0, 0] = 1; W_loc[3, 0] = 1  # c00 (idx 0), c11 (idx 3)
    
    # M2 = (a00+a10)*b11 -> c01 += M2, c11 -= M2
    U_loc[0, 1] = 1; U_loc[2, 1] = 1
    V_loc[1, 1] = 1
    W_loc[1, 1] = 1; W_loc[3, 1] = -1  # c01 (idx 1), c11 (idx 3)
    
    # M3 = a00*(b01-b11) -> c01 += M3, c11 += M3
    U_loc[0, 2] = 1
    V_loc[1, 2] = 1; V_loc[3, 2] = -1
    W_loc[1, 2] = 1; W_loc[3, 2] = 1  # c01 (idx 1), c11 (idx 3)
    
    # M4 = a10*b11 -> c00 += M4, c10 += M4
    U_loc[2, 3] = 1
    V_loc[3, 3] = 1; V_loc[1, 3] = -1
    W_loc[0, 3] = 1; W_loc[2, 3] = 1  # c00 (idx 0), c10 (idx 2)
    
    # M5 = (a01+a11)*b00 -> c00 -= M5, c01 += M5
    U_loc[1, 4] = 1; U_loc[3, 4] = 1
    V_loc[0, 4] = 1
    W_loc[0, 4] = -1; W_loc[1, 4] = 1  # c00 (idx 0), c01 (idx 1)
    
    # M6 = (a01-a10)*(b00+b10) -> c11 += M6
    U_loc[1, 5] = 1; U_loc[2, 5] = -1
    V_loc[0, 5] = 1; V_loc[2, 5] = 1
    W_loc[3, 5] = 1  # c11 (idx 3)
    
    # M7 = (a11-a00)*(b00+b01) -> c00 += M7
    U_loc[3, 6] = 1; U_loc[0, 6] = -1
    V_loc[0, 6] = 1; V_loc[1, 6] = 1
    W_loc[0, 6] = 1  # c00 (idx 0)
    
    # Extra part (cols 7-10) for Row 2 of A (a20, a21)
    # M8 = a20 * b00 -> c20
    U_loc[4, 7] = 1
    V_loc[0, 7] = 1
    W_loc[4, 7] = 1  # c20 (idx 4)
    
    # M9 = a21 * b10 -> c20
    U_loc[5, 8] = 1
    V_loc[2, 8] = 1
    W_loc[4, 8] = 1  # c20 (idx 4)
    
    # M10 = a20 * b01 -> c21
    U_loc[4, 9] = 1
    V_loc[1, 9] = 1
    W_loc[5, 9] = 1  # c21 (idx 5)
    
    # M11 = a21 * b11 -> c21
    U_loc[5, 10] = 1
    V_loc[3, 10] = 1
    W_loc[5, 10] = 1  # c21 (idx 5)
    
    # Mapping function
    def add_block(block_idx, a_col_start, b_row_start, b_col_start, c_col_start):
        col_offset = block_idx * 11
        
        # A indices: rows 0-2, cols a_col_start..a_col_start+1
        for r in range(3):
            for c in range(2):
                local_a_idx = r * 2 + c
                global_a_idx = r * m + (a_col_start + c)
                U[global_a_idx, col_offset:col_offset+11] += U_loc[local_a_idx, :]
        
        # B indices: rows b_row_start..b_row_start+1, cols b_col_start..b_col_start+1
        for r in range(2):
            for c in range(2):
                local_b_idx = r * 2 + c
                global_b_idx = (b_row_start + r) * p + (b_col_start + c)
                V[global_b_idx, col_offset:col_offset+11] += V_loc[local_b_idx, :]
        
        # C indices: rows 0-2, cols c_col_start..c_col_start+1
        # W index: col * n + row
        for r in range(3):
            for c in range(2):
                local_c_idx = c * 3 + r  # Wait, need to match local C indexing
                # Local C: c00=0, c01=1, c10=2, c11=3, c20=4, c21=5
                # That's row-major: r*2+c
                local_c_idx = r * 2 + c
                global_c_idx = (c_col_start + c) * n + r
                W[global_c_idx, col_offset:col_offset+11] += W_loc[local_c_idx, :]
    
    # Block 0: A(cols 0,1) * B(rows 0,1; cols 0,1) -> C(cols 0,1)
    add_block(0, 0, 0, 0, 0)
    
    # Block 1: A(cols 2,3) * B(rows 2,3; cols 0,1) -> C(cols 0,1)
    add_block(1, 2, 2, 0, 0)
    
    # Block 2: A(cols 0,1) * B(rows 0,1; cols 2,3) -> C(cols 2,3)
    add_block(2, 0, 0, 2, 2)
    
    # Block 3: A(cols 2,3) * B(rows 2,3; cols 2,3) -> C(cols 2,3)
    add_block(3, 2, 2, 2, 2)
    
    # Check deadline
    elapsed = time.time() - start_time
    if elapsed > deadline:
        return {
            "U": U.tolist(),
            "V": V.tolist(),
            "W": W.tolist()
        }
    
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
