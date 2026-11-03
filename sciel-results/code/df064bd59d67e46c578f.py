import numpy as np
import time

def construct():
    """
    Constructs a rank-44 tensor decomposition for 3x4 * 4x4 matrix multiplication.
    
    The decomposition uses a block strategy:
    - Splits the 3x4 matrix A into a 2x4 top part and a 1x4 bottom part.
    - Splits the 4x4 matrix B into four 2x2 blocks.
    - Uses Strassen's rank-7 algorithm for the 2x2 * 2x2 -> 2x2 blocks (4 blocks).
    - Uses trivial rank-4 algorithm for the 1x2 * 2x2 -> 1x2 blocks (4 blocks).
    - Total rank: 4 * 7 + 4 * 4 = 28 + 16 = 44.
    """
    n, m, p = 3, 4, 4
    
    # Shapes
    dim_u = n * m  # 12
    dim_v = m * p  # 16
    dim_w = p * n  # 12
    
    rank = 44
    U = np.zeros((dim_u, rank))
    V = np.zeros((dim_v, rank))
    W = np.zeros((dim_w, rank))
    
    # Local factors for 2x2x2 Strassen (rank 7)
    # Indices for 2x2:
    # A: 0,1 (row 0), 2,3 (row 1) -> Flat: 0,1,2,3
    # B: 0,1 (col 0), 2,3 (col 1) -> Flat: 0,1,2,3 (Row-major: 0,1 are col 0? No. B is 2x2.
    # Let's define local indices carefully.
    # Local A (2x2): indices 0,1,2,3. 0=(0,0), 1=(0,1), 2=(1,0), 3=(1,1)
    # Local B (2x2): indices 0,1,2,3. 0=(0,0), 1=(1,0), 2=(0,1), 3=(1,1) ??
    # Standard flattening:
    # A[i,j] -> i*2+j
    # B[i,j] -> i*2+j
    # C[i,j] -> j*2+i (Column major for C? The global W is column major.
    # Let's check the global mapping in the seed:
    # c_col, c_row = divmod(c, n). So c = c_col * n + c_row.
    # For local 2x2 C, if we map to global, we need to be consistent.
    
    # Let's define the local Strassen factors for 2x2x2.
    # U_loc (4, 7), V_loc (4, 7), W_loc (4, 7)
    # Local A index: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
    # Local B index: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
    # Local C index: 0:(0,0), 1:(1,0), 2:(0,1), 3:(1,1) -- Column major?
    # If global C is column major (col * n + row), then for local 2x2 block at offset (r_off, c_off):
    # Global C index = (c_off + c_col) * 3 + (r_off + c_row)
    # This mapping is handled by the add_block function.
    
    U_loc = np.zeros((4, 7))
    V_loc = np.zeros((4, 7))
    W_loc = np.zeros((4, 7))
    
    # Strassen Factors for 2x2
    # M1 = (A00+A10)(B00+B10)
    U_loc[0, 0] = 1; U_loc[2, 0] = 1
    V_loc[0, 0] = 1; V_loc[2, 0] = 1
    W_loc[0, 0] = 1; W_loc[1, 0] = 1; W_loc[2, 0] = 1; W_loc[3, 0] = 1
    
    # M2 = A10(B00+B10)
    U_loc[2, 1] = 1
    V_loc[0, 1] = 1; V_loc[2, 1] = 1
    W_loc[1, 1] = 1; W_loc[3, 1] = 1
    
    # M3 = A00(B10-B00)
    U_loc[0, 2] = 1
    V_loc[0, 2] = -1; V_loc[2, 2] = 1
    W_loc[0, 2] = 1; W_loc[2, 2] = 1
    
    # M4 = (A01+A11)(B00+B10) -- Wait, standard Strassen:
    # M4 = (A01+A11)(B00+B10) is not standard.
    # Standard:
    # M1 = (A00+A10)(B00+B10)
    # M2 = (A00+A10)(B10-B00)
    # M3 = A00(B10-B00)
    # M4 = A11(B00+B10)
    # M5 = (A00+A01)(B11-B01)
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    
    # Let's use the standard set from Wikipedia/Textbooks.
    # C00 = M1 - M2 + M3 - M4
    # C10 = M2 + M4
    # C01 = M5 + M6
    # C11 = M1 - M3 + M5 - M6 + M7 ? No.
    
    # Let's use the specific factors from the seed code which were verified.
    # Seed U_loc:
    # Col 0: U[0]=1, U[3]=1 (A00, A11? No, seed used 0,3 for A00,A11?
    # Seed: U_loc[0, 0]=1; U_loc[3, 0]=1.
    # Seed indices: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1).
    # So Seed Col 0 uses A00, A11.
    
    # I will copy the verified factors from the seed to ensure correctness.
    U_loc = np.array([
        [1, 0, 1, 0, 1, -1, 0],
        [0, 0, 0, 0, 1, 0, 1],
        [0, 1, 0, 0, 0, 1, 0],
        [1, 1, 0, 1, 0, 0, -1]
    ], dtype=float)
    
    V_loc = np.array([
        [1, 1, 1, -1, 0, 1, 1],
        [0, 0, 1, 0, 1, 1, 0],
        [1, 0, 0, 1, 0, 0, 1],
        [1, 1, -1, 0, 0, 1, 1]
    ], dtype=float)
    
    W_loc = np.array([
        [1, 0, 1, 1, -1, 0, 1],
        [0, 1, 0, 0, 1, 0, 0],
        [0, 0, 1, 1, 0, 1, 0],
        [1, -1, 1, 0, 0, 0, 0]
    ], dtype=float)
    
    # Verify local Strassen
    # A is 2x2, B is 2x2, C is 2x2.
    # C = A B.
    # Indices:
    # a: 0(0,0), 1(0,1), 2(1,0), 3(1,1)
    # b: 0(0,0), 1(0,1), 2(1,0), 3(1,1)
    # c: 0(0,0), 1(1,0), 2(0,1), 3(1,1) -- Column major?
    # In seed: c_col, c_row = divmod(c, n).
    # For local 2x2, if we assume column major:
    # c=0 -> (0,0). c=1 -> (0,1)? No.
    # divmod(1, 2) -> 0, 1. col=0, row=1. So c=1 is (row 1, col 0).
    # So local C indices: 0:(0,0), 1:(1,0), 2:(0,1), 3:(1,1).
    
    for a in range(4):
        a_r, a_c = divmod(a, 2)
        for b in range(4):
            b_r, b_c = divmod(b, 2)
            for c in range(4):
                c_c, c_r = divmod(c, 2)
                expected = 1.0 if (a_c == b_r and c_r == a_r and c_c == b_c) else 0.0
                computed = np.sum(U_loc[a, :] * V_loc[b, :] * W_loc[c, :])
                if abs(computed - expected) > 1e-10:
                    raise ValueError(f"Local Strassen error at ({a},{b},{c}): {computed} != {expected}")

    # Local factors for 1x2x2 (Rank 4)
    # A is 1x2 (indices 0,1), B is 2x2 (indices 0,1,2,3), C is 1x2 (indices 0,1)
    # C[0] = A[0]B[0,0] + A[1]B[1,0]
    # C[1] = A[0]B[0,1] + A[1]B[1,1]
    U_1x2 = np.array([
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ], dtype=float) # Wait, A has 2 elements. U should be (2, 4).
    
    # A indices: 0, 1.
    # B indices: 0(0,0), 1(0,1), 2(1,0), 3(1,1).
    # C indices: 0(0,0), 1(0,1) -- Row major for 1x2?
    # Global C is column major. For 1x2 block, rows=1, cols=2.
    # Global c = c_col * 3 + c_row.
    # If local c is 0 or 1, and it maps to global c_row=0, c_col=0 or 1.
    # So local C index 0 -> (row 0, col 0). Local C index 1 -> (row 0, col 1).
    
    U_1x2 = np.zeros((2, 4))
    V_1x2 = np.zeros((4, 4))
    W_1x2 = np.zeros((2, 4))
    
    # Term 1: A0 * B00 -> C0
    U_1x2[0, 0] = 1
    V_1x2[0, 0] = 1
    W_1x2[0, 0] = 1
    
    # Term 2: A1 * B10 -> C0
    U_1x2[1, 1] = 1
    V_1x2[2, 1] = 1
    W_1x2[0, 1] = 1
    
    # Term 3: A0 * B01 -> C1
    U_1x2[0, 2] = 1
    V_1x2[1, 2] = 1
    W_1x2[1, 2] = 1
    
    # Term 4: A1 * B11 -> C1
    U_1x2[1, 3] = 1
    V_1x2[3, 3] = 1
    W_1x2[1, 3] = 1
    
    def add_block_2x2x2(block_idx, a_row_start, a_col_start, b_row_start, b_col_start, c_row_start, c_col_start):
        col_offset = block_idx * 7
        for r in range(2):
            for c in range(2):
                local_a = r * 2 + c
                global_a_row = a_row_start + r
                global_a_col = a_col_start + c
                global_a_idx = global_a_row * 4 + global_a_col
                U[global_a_idx, col_offset:col_offset+7] += U_loc[local_a, :]
        
        for r in range(2):
            for c in range(2):
                local_b = r * 2 + c
                global_b_row = b_row_start + r
                global_b_col = b_col_start + c
                global_b_idx = global_b_row * 4 + global_b_col
                V[global_b_idx, col_offset:col_offset+7] += V_loc[local_b, :]
        
        for r in range(2):
            for c in range(2):
                local_c = c * 2 + r # Column major local?
                # Local C index in W_loc is defined as:
                # 0:(0,0), 1:(1,0), 2:(0,1), 3:(1,1).
                # This is col * 2 + row.
                global_c_row = c_row_start + r
                global_c_col = c_col_start + c
                global_c_idx = global_c_col * 3 + global_c_row
                W[global_c_idx, col_offset:col_offset+7] += W_loc[local_c, :]

    def add_block_1x2x2(block_idx, a_row_start, a_col_start, b_row_start, b_col_start, c_row_start, c_col_start):
        col_offset = 28 + block_idx * 4
        for c in range(2):
            local_a = c
            global_a_row = a_row_start
            global_a_col = a_col_start + c
            global_a_idx = global_a_row * 4 + global_a_col
            U[global_a_idx, col_offset:col_offset+4] += U_1x2[local_a, :]
        
        for r in range(2):
            for c in range(2):
                local_b = r * 2 + c
                global_b_row = b_row_start + r
                global_b_col = b_col_start + c
                global_b_idx = global_b_row * 4 + global_b_col
                V[global_b_idx, col_offset:col_offset+4] += V_1x2[local_b, :]
        
        for c in range(2):
            local_c = c
            global_c_row = c_row_start
            global_c_col = c_col_start + c
            global_c_idx = global_c_col * 3 + global_c_row
            W[global_c_idx, col_offset:col_offset+4] += W_1x2[local_c, :]

    # Add 4 Strassen blocks (Top 2 rows of A)
    add_block_2x2x2(0, 0, 0, 0, 0, 0, 0)
    add_block_2x2x2(1, 0, 2, 2, 0, 0, 0)
    add_block_2x2x2(2, 0, 0, 0, 2, 0, 2)
    add_block_2x2x2(3, 0, 2, 2, 2, 0, 2)
    
    # Add 4 Trivial blocks (Bottom 1 row of A)
    add_block_1x2x2(0, 2, 0, 0, 0, 2, 0)
    add_block_1x2x2(1, 2, 2, 2, 0, 2, 0)
    add_block_1x2x2(2, 2, 0, 0, 2, 2, 2)
    add_block_1x2x2(3, 2, 2, 2, 2, 2, 2)
    
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
