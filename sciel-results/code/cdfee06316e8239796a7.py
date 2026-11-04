import numpy as np
from fractions import Fraction

def construct():
    # 2x4 by 4x8 matrix multiplication: rank-56 decomposition
    # A: 2x4, B: 4x8, C: 2x8
    #
    # Strategy:
    # 1. Split A into 2 row blocks (each 1x4) and B into 2 row blocks (each 2x8)
    #    and 2 col blocks (each 4x4).
    #    Actually, let's use the approach from the current program but restructured.
    #
    # The current program uses 2x2 Strassen on the intermediate dimension m=4.
    # For each of the 4 column-pairs j (in the 8-wide output), and for each of the
    # 2 row-blocks s of B, we do a 2x2x2 Strassen (rank 7).
    # Total: 4 * 2 * 7 = 56.
    #
    # I'll implement this cleanly using the Strassen formulas.
    
    n, m, p = 2, 4, 8
    total_rank = 56
    
    U = np.zeros((n * m, total_rank), dtype=object)  # 8 x 56
    V = np.zeros((m * p, total_rank), dtype=object)  # 32 x 56
    W = np.zeros((p * n, total_rank), dtype=object)  # 16 x 56
    
    # Strassen 2x2 formulas (indices for local 2x2 blocks):
    # A = [[A00, A01], [A10, A11]]
    # B = [[B00, B01], [B10, B11]]
    # C = A * B
    #
    # M1 = (A00 + A11) * (B00 + B11)
    # M2 = (A10 + A11) * B00
    # M3 = A00 * (B01 - B11)
    # M4 = A11 * (B10 - B00)
    # M5 = (A00 + A01) * B11
    # M6 = (A10 - A00) * (B00 + B01)
    # M7 = (A01 - A11) * (B10 + B11)
    #
    # C00 = M1 - M4 + M5 - M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # For our problem:
    # A is 2x4. We think of it as two 1x2 blocks per row? No.
    # Let me reconsider the block structure.
    #
    # A: 2x4. Split columns into two groups of 2: A = [A_left | A_right]
    #   A_left: 2x2, A_right: 2x2
    # B: 4x8. Split rows into two groups of 2: B = [B_top; B_bottom]
    #   B_top: 2x8, B_bottom: 2x8
    #   Split columns of B_top and B_bottom into two groups of 4.
    #
    # C = A * B = A_left * B_top + A_right * B_bottom
    #
    # For A_left * B_top: 2x2 * 2x8
    # For A_right * B_bottom: 2x2 * 2x8
    #
    # Each 2x2 * 2x8 can be seen as:
    #   A_2x2 * B_2x8 where B_2x8 has 8 columns.
    #   Split B_2x8 into 4 blocks of 2x2.
    #   For each 2x2 block, use Strassen (rank 7).
    #   4 blocks * rank 7 = rank 28 per A_2x2 * B_2x8.
    #   Two of these (left and right) = 56 total.
    
    # Strassen local coefficient tables
    # For a 2x2x2 multiplication:
    # U_local[i, k] for A, V_local[j, k] for B, W_local[i, k] for C
    # where i,j are 0..3 (row*2+col), k is 0..6 (the 7 products)
    
    U_loc = np.zeros((4, 7), dtype=object)
    V_loc = np.zeros((4, 7), dtype=object)
    W_loc = np.zeros((4, 7), dtype=object)
    
    # M1: (A00+A11)(B00+B11) -> C00 += 1, C11 += 1
    U_loc[0, 0] += 1; U_loc[3, 0] += 1
    V_loc[0, 0] += 1; V_loc[3, 0] += 1
    W_loc[0, 0] += 1; W_loc[3, 0] += 1
    
    # M2: (A10+A11)B00 -> C10 += 1, C11 += -1
    U_loc[2, 1] += 1; U_loc[3, 1] += 1
    V_loc[0, 1] += 1
    W_loc[2, 1] += 1; W_loc[3, 1] += -1
    
    # M3: A00(B01-B11) -> C01 += 1, C11 += 1
    U_loc[0, 2] += 1
    V_loc[1, 2] += 1; V_loc[3, 2] += -1
    W_loc[1, 2] += 1; W_loc[3, 2] += 1
    
    # M4: A11(B10-B00) -> C00 += 1, C10 += 1
    U_loc[3, 3] += 1
    V_loc[2, 3] += 1; V_loc[0, 3] += -1
    W_loc[0, 3] += 1; W_loc[2, 3] += 1
    
    # M5: (A00+A01)B11 -> C00 += -1, C01 += 1
    U_loc[0, 4] += 1; U_loc[1, 4] += 1
    V_loc[3, 4] += 1
    W_loc[0, 4] += -1; W_loc[1, 4] += 1
    
    # M6: (A10-A00)(B00+B01) -> C11 += 1
    U_loc[2, 5] += 1; U_loc[0, 5] += -1
    V_loc[0, 5] += 1; V_loc[1, 5] += 1
    W_loc[3, 5] += 1
    
    # M7: (A01-A11)(B10+B11) -> C00 += 1
    U_loc[1, 6] += 1; U_loc[3, 6] += -1
    V_loc[2, 6] += 1; V_loc[3, 6] += 1
    W_loc[0, 6] += 1
    
    # Now build the full decomposition
    # A: 2x4, index a = row*4 + col (0..7)
    # B: 4x8, index b = row*8 + col (0..31)
    # C: 2x8, index c = col*2 + row (0..15) [trace convention: W[k*n+i, r] where k=col, i=row]
    
    # Block structure:
    # A_left = A[:, 0:2] (rows 0,1; cols 0,1)
    # A_right = A[:, 2:4] (rows 0,1; cols 2,3)
    # B_top = B[0:2, :] (rows 0,1; cols 0..7)
    # B_bottom = B[2:4, :] (rows 2,3; cols 0..7)
    
    # For A_left * B_top:
    #   A_left is 2x2: A[0,0], A[0,1], A[1,0], A[1,1]
    #   B_top is 2x8: split into 4 blocks of 2x2:
    #     B_top[:, 0:2], B_top[:, 2:4], B_top[:, 4:6], B_top[:, 6:8]
    
    # For A_right * B_bottom:
    #   A_right is 2x2: A[0,2], A[0,3], A[1,2], A[1,3]
    #   B_bottom is 2x8: split into 4 blocks of 2x2:
    #     B_bottom[:, 0:2], B_bottom[:, 2:4], B_bottom[:, 4:6], B_bottom[:, 6:8]
    
    # Map local indices to global:
    # For A_left (2x2): local (r,c) -> global a = r*4 + c
    #   local 0=(0,0)->0, 1=(0,1)->1, 2=(1,0)->4, 3=(1,1)->5
    # For A_right (2x2): local (r,c) -> global a = r*4 + (2+c)
    #   local 0=(0,0)->2, 1=(0,1)->3, 2=(1,0)->6, 3=(1,1)->7
    
    # For B_top (2x8): local block j (0..3), within block (r,c) -> global b = r*8 + (2*j+c)
    #   local 0=(0,0), 1=(0,1), 2=(1,0), 3=(1,1)
    #   For block j: b0 = 0*8+2j, b1 = 0*8+2j+1, b2 = 1*8+2j, b3 = 1*8+2j+1
    
    # For B_bottom (2x8): local block j, within block (r,c) -> global b = (2+r)*8 + (2*j+c)
    #   local 0=(0,0), 1=(0,1), 2=(1,0), 3=(1,1)
    #   For block j: b0 = 2*8+2j, b1 = 2*8+2j+1, b2 = 3*8+2j, b3 = 3*8+2j+1
    
    # For C: local (row, col_pair) -> global c = (2*col_pair + c_local)*2 + row
    #   Wait, C is 2x8. c index = col*2 + row (trace convention: W[k*n+i] where k=col, n=2, i=row)
    #   For a 2x2 output block at column pair j (cols 2j, 2j+1):
    #     local 0=(row0, col0) -> c = (2j)*2 + 0 = 4j
    #     local 1=(row0, col1) -> c = (2j+1)*2 + 0 = 4j+2
    #     local 2=(row1, col0) -> c = (2j)*2 + 1 = 4j+1
    #     local 3=(row1, col1) -> c = (2j+1)*2 + 1 = 4j+3
    
    a_left_map = [0, 1, 4, 5]
    a_right_map = [2, 3, 6, 7]
    
    for part in range(2):  # 0=left, 1=right
        a_map = a_left_map if part == 0 else a_right_map
        b_row_offset = 0 if part == 0 else 2  # B_top starts at row 0, B_bottom at row 2
        
        for j in range(4):  # column pair
            base_rank = (part * 4 + j) * 7
            
            # B indices for this 2x2 block
            b_start_col = 2 * j
            b_map = [
                (b_row_offset + 0) * 8 + b_start_col,
                (b_row_offset + 0) * 8 + b_start_col + 1,
                (b_row_offset + 1) * 8 + b_start_col,
                (b_row_offset + 1) * 8 + b_start_col + 1
            ]
            
            # C indices for this 2x2 block
            c_start_col = 2 * j
            c_map = [
                c_start_col * 2 + 0,
                (c_start_col + 1) * 2 + 0,
                c_start_col * 2 + 1,
                (c_start_col + 1) * 2 + 1
            ]
            
            for k in range(7):
                r = base_rank + k
                for la in range(4):
                    val = U_loc[la, k]
                    if val != 0:
                        U[a_map[la], r] += val
                for lb in range(4):
                    val = V_loc[lb, k]
                    if val != 0:
                        V[b_map[lb], r] += val
                for lc in range(4):
                    val = W_loc[lc, k]
                    if val != 0:
                        W[c_map[lc], r] += val
    
    # Convert to int lists
    U_list = [[int(U[i, j]) for j in range(total_rank)] for i in range(n * m)]
    V_list = [[int(V[i, j]) for j in range(total_rank)] for i in range(m * p)]
    W_list = [[int(W[i, j]) for j in range(total_rank)] for i in range(p * n)]
    
    return {"U": U_list, "V": V_list, "W": W_list}
