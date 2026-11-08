import numpy as np
import time

def construct():
    n, m, p = 3, 4, 4
    deadline = time.time() + 140.0
    
    # We build a rank-44 decomposition using a 3x2x2 local factor.
    # The 3x2x2 factor has rank 11 (Strassen 7 + 4 extra for row 2).
    # 4 blocks * 11 = 44.
    
    rank = 44
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    # Local 3x2x2 factors: A is 3x2, B is 2x2, C is 3x2
    # Indices: A[r*2+c], B[r*2+c], C[r*2+c] for local blocks
    # Global C index: W[k*n + r, t] where k is C col, r is C row
    
    U_loc = np.zeros((6, 11))
    V_loc = np.zeros((4, 11))
    W_loc = np.zeros((6, 11))
    
    # Strassen for 2x2x2 (rows 0-1 of A, all of B, rows 0-1 of C)
    # M1: (a00+a10)(b00+b11) -> c00 += M1, c11 += M1
    U_loc[0, 0] = 1; U_loc[3, 0] = 1
    V_loc[0, 0] = 1; V_loc[3, 0] = 1
    W_loc[0, 0] = 1; W_loc[3, 0] = 1
    
    # M2: (a10)(b00) -> c10 += M2, c11 -= M2
    U_loc[2, 1] = 1; U_loc[3, 1] = 1
    V_loc[0, 1] = 1
    W_loc[2, 1] = 1; W_loc[3, 1] = -1
    
    # M3: (a00)(b11-b01) -> c01 += M3, c11 += M3
    U_loc[0, 2] = 1
    V_loc[1, 2] = 1; V_loc[3, 2] = -1
    W_loc[1, 2] = 1; W_loc[3, 2] = 1
    
    # M4: (a10)(b10-b00) -> c00 += M4, c10 += M4
    U_loc[3, 3] = 1
    V_loc[2, 3] = 1; V_loc[0, 3] = -1
    W_loc[0, 3] = 1; W_loc[2, 3] = 1
    
    # M5: (a00+a01)(b10) -> c00 -= M5, c01 += M5
    U_loc[0, 4] = 1; U_loc[1, 4] = 1
    V_loc[3, 4] = 1
    W_loc[0, 4] = -1; W_loc[1, 4] = 1
    
    # M6: (a10-a00)(b00+b01) -> c11 += M6
    U_loc[2, 5] = 1; U_loc[0, 5] = -1
    V_loc[0, 5] = 1; V_loc[1, 5] = 1
    W_loc[3, 5] = 1
    
    # M7: (a01-a10)(b10+b11) -> c00 += M7
    U_loc[1, 6] = 1; U_loc[3, 6] = -1
    V_loc[2, 6] = 1; V_loc[3, 6] = 1
    W_loc[0, 6] = 1
    
    # Extra terms for row 2 of A (3rd row)
    # M8: a20 * b00 -> c20
    U_loc[4, 7] = 1
    V_loc[0, 7] = 1
    W_loc[4, 7] = 1
    
    # M9: a21 * b10 -> c20
    U_loc[5, 8] = 1
    V_loc[2, 8] = 1
    W_loc[4, 8] = 1
    
    # M10: a20 * b01 -> c21
    U_loc[4, 9] = 1
    V_loc[1, 9] = 1
    W_loc[5, 9] = 1
    
    # M11: a21 * b11 -> c21
    U_loc[5, 10] = 1
    V_loc[3, 10] = 1
    W_loc[5, 10] = 1
    
    # Place 4 blocks into the global 3x4x4 decomposition
    # Block (bi, bj): A block i (cols 2i..2i+1) * B block j (rows 2j..2j+1, cols 2j..2j+1) -> C block (i,j)
    # Wait, the 3x4 matrix A has 4 columns. We split into 2 blocks of 2 cols each.
    # The 4x4 matrix B is split into 2x2 blocks: B[j, k] where j is row-block, k is col-block.
    # C[i, j] = sum over k of A[i, k] * B[k, j]
    # So C block (i, j) = sum_k A_col_block_i * B_row_block_k, col_block_j
    
    # Actually let me re-examine. A is 3x4, B is 4x4, C is 3x4.
    # C[i, j] = sum_k A[i, k] * B[k, j]
    # Split A columns into blocks: A[:, 0:2] and A[:, 2:4]
    # Split B rows into blocks: B[0:2, :] and B[2:4, :]
    # Split B columns into blocks: B[:, 0:2] and B[:, 2:4]
    # C[:, 0:2] = A[:, 0:2] @ B[0:2, 0:2] + A[:, 2:4] @ B[2:4, 0:2]
    # C[:, 2:4] = A[:, 0:2] @ B[0:2, 2:4] + A[:, 2:4] @ B[2:4, 2:4]
    
    # So we have 4 block multiplications, each is 3x2 times 2x2 gives 3x2.
    # Each uses the local 3x2x2 factor with rank 11.
    
    def place_block(block_idx, a_col_offset, b_row_offset, b_col_offset, c_col_offset):
        """Place one 3x2x2 block multiplication.
        a_col_offset: which 2 columns of A (0 or 2)
        b_row_offset: which 2 rows of B (0 or 2)
        b_col_offset: which 2 columns of B (0 or 2)
        c_col_offset: which 2 columns of C (0 or 2)
        """
        base = block_idx * 11
        
        for r in range(3):
            for c in range(2):
                local_a = r * 2 + c
                global_a = r * m + (a_col_offset + c)
                U[global_a, base:base+11] = U_loc[local_a, :]
        
        for r in range(2):
            for c in range(2):
                local_b = r * 2 + c
                global_b = (b_row_offset + r) * p + (b_col_offset + c)
                V[global_b, base:base+11] = V_loc[local_b, :]
        
        for r in range(3):
            for c in range(2):
                local_c = r * 2 + c
                # C index: W[k*n + r, t] where k = C col, r = C row
                global_c = (c_col_offset + c) * n + r
                W[global_c, base:base+11] = W_loc[local_c, :]
    
    # Block 0: A[:, 0:2] * B[0:2, 0:2] -> C[:, 0:2]
    place_block(0, 0, 0, 0, 0)
    
    # Block 1: A[:, 2:4] * B[2:4, 0:2] -> C[:, 0:2]
    place_block(1, 2, 2, 0, 0)
    
    # Block 2: A[:, 0:2] * B[0:2, 2:4] -> C[:, 2:4]
    place_block(2, 0, 0, 2, 2)
    
    # Block 3: A[:, 2:4] * B[2:4, 2:4] -> C[:, 2:4]
    place_block(3, 2, 2, 2, 2)
    
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
