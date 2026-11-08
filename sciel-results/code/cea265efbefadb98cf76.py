import numpy as np

def construct():
    n, m, p = 3, 4, 4
    
    # Rank of 3x2x2 block is 11
    # Total rank = 4 * 11 = 44
    rank = 44
    
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    # Define the rank 11 decomposition for 3x2x2
    # U_loc: (3*2, 11) = (6, 11)
    # V_loc: (2*2, 11) = (4, 11)
    # W_loc: (2*3, 11) = (6, 11) -- Note: W index is k*n + i = k*3 + i
    U_loc = np.zeros((6, 11))
    V_loc = np.zeros((4, 11))
    W_loc = np.zeros((6, 11))
    
    # Based on the current program's values
    U_loc[0, 0] = 1; U_loc[3, 0] = 1
    V_loc[0, 0] = 1; V_loc[3, 0] = 1
    W_loc[0, 0] = 1; W_loc[3, 0] = 1
    
    U_loc[2, 1] = 1; U_loc[3, 1] = 1
    V_loc[0, 1] = 1
    W_loc[2, 1] = 1; W_loc[3, 1] = -1
    
    U_loc[0, 2] = 1
    V_loc[1, 2] = 1; V_loc[3, 2] = -1
    W_loc[1, 2] = 1; W_loc[3, 2] = 1
    
    U_loc[3, 3] = 1
    V_loc[2, 3] = 1; V_loc[0, 3] = -1
    W_loc[0, 3] = 1; W_loc[2, 3] = 1
    
    U_loc[0, 4] = 1; U_loc[1, 4] = 1
    V_loc[3, 4] = 1
    W_loc[0, 4] = -1; W_loc[1, 4] = 1
    
    U_loc[2, 5] = 1; U_loc[0, 5] = -1
    V_loc[0, 5] = 1; V_loc[1, 5] = 1
    W_loc[3, 5] = 1
    
    U_loc[1, 6] = 1; U_loc[3, 6] = -1
    V_loc[2, 6] = 1; V_loc[3, 6] = 1
    W_loc[0, 6] = 1
    
    U_loc[4, 7] = 1
    V_loc[0, 7] = 1
    W_loc[4, 7] = 1
    
    U_loc[5, 8] = 1
    V_loc[2, 8] = 1
    W_loc[4, 8] = 1
    
    U_loc[4, 9] = 1
    V_loc[1, 9] = 1
    W_loc[5, 9] = 1
    
    U_loc[5, 10] = 1
    V_loc[3, 10] = 1
    W_loc[5, 10] = 1
    
    def add_block(block_idx, a_row_start, a_col_start, b_row_start, b_col_start, c_row_start, c_col_start):
        col_offset = block_idx * 11
        for r in range(3):
            for c in range(2):
                local_a_idx = r * 2 + c
                global_a_row = a_row_start + r
                global_a_col = a_col_start + c
                global_a_idx = global_a_row * m + global_a_col
                U[global_a_idx, col_offset:col_offset+11] += U_loc[local_a_idx, :]
        for r in range(2):
            for c in range(2):
                local_b_idx = r * 2 + c
                global_b_row = b_row_start + r
                global_b_col = b_col_start + c
                global_b_idx = global_b_row * p + global_b_col
                V[global_b_idx, col_offset:col_offset+11] += V_loc[local_b_idx, :]
        for r in range(3):
            for c in range(2):
                local_c_idx = r * 2 + c
                global_c_row = c_row_start + r
                global_c_col = c_col_start + c
                global_c_idx = global_c_col * n + global_c_row
                W[global_c_idx, col_offset:col_offset+11] += W_loc[local_c_idx, :]

    # Block 0: A[0:3, 0:2] * B[0:2, 0:2] -> C[0:3, 0:2]
    add_block(0, 0, 0, 0, 0, 0, 0)
    # Block 1: A[0:3, 2:4] * B[0:2, 2:4] -> C[0:3, 2:4]
    # Wait, B is 4x4.
    # C_{i,k} = sum_j A_{i,j} B_{j,k}
    # If we split j into 0,1 and 2,3.
    # Term 1: j in 0,1. B rows 0,1.
    # Term 2: j in 2,3. B rows 2,3.
    # So Block 0 uses B rows 0,1.
    # Block 1 uses B rows 2,3.
    # And both use all columns of C?
    # No, the block is 3x2x2.
    # So it computes C for 2 columns.
    # We need to cover 4 columns of C.
    # So we need 2 blocks for each j-split?
    # Yes.
    # j in 0,1:
    #   k in 0,1: Block 0
    #   k in 2,3: Block 1
    # j in 2,3:
    #   k in 0,1: Block 2
    #   k in 2,3: Block 3
    
    # Block 1: A[0:3, 0:2] * B[0:2, 2:4] -> C[0:3, 2:4]
    add_block(1, 0, 0, 0, 2, 0, 2)
    
    # Block 2: A[0:3, 2:4] * B[2:4, 0:2] -> C[0:3, 0:2]
    add_block(2, 0, 2, 2, 0, 0, 0)
    
    # Block 3: A[0:3, 2:4] * B[2:4, 2:4] -> C[0:3, 2:4]
    add_block(3, 0, 2, 2, 2, 0, 2)

    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
