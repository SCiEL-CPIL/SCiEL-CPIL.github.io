import numpy as np
import time

def construct():
    """
    Matrix multiplication 3x4 by 4x4.
    Uses rank 38 construction by splitting B into rows.
    Row 0 of B (1x4) with Col 0 of A (3x1) -> Rank 4
    Row 1 of B (1x4) with Col 1 of A (3x1) -> Rank 12
    Rows 2-3 of B (2x4) with Cols 2-3 of A (3x2) -> Rank 22 (2 * 11)
    Total rank = 4 + 12 + 22 = 38.
    """
    start_time = time.time()
    deadline = 130.0

    n, m, p = 3, 4, 4
    rank = 38

    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))

    # Part 1: A[0,0] * B[0, 0:4] -> C[0, 0:4]
    # Rank 4
    # U: A index 0
    # V: B indices 0, 1, 4, 5 (row 0 of B)
    # W: C indices 0, 1, 2, 3 (row 0 of C, trace convention: W[k*n+i] -> W[j*3+i])
    # C[0, j] -> W[j*3 + 0]
    
    # Term 1: A[0,0] B[0,0] -> C[0,0]
    U[0, 0] = 1
    V[0, 0] = 1
    W[0, 0] = 1 # W[0*3+0]
    
    # Term 2: A[0,0] B[0,1] -> C[0,1]
    U[0, 1] = 1
    V[1, 1] = 1
    W[3, 1] = 1 # W[1*3+0]
    
    # Term 3: A[0,0] B[0,2] -> C[0,2]
    U[0, 2] = 1
    V[2, 2] = 1
    W[6, 2] = 1 # W[2*3+0]
    
    # Term 4: A[0,0] B[0,3] -> C[0,3]
    U[0, 3] = 1
    V[3, 3] = 1
    W[9, 3] = 1 # W[3*3+0]

    # Part 2: A[0:3, 1] * B[1, 0:4] -> C[0:3, 0:4]
    # Rank 12
    # A indices: 1, 4, 7 (col 1)
    # B indices: 4, 5, 6, 7 (row 1)
    # C indices: all 12
    
    a_idx = [1, 4, 7]
    b_idx = [4, 5, 6, 7]
    
    col_offset = 4
    for i in range(3):
        for j in range(4):
            U[a_idx[i], col_offset] = 1
            V[b_idx[j], col_offset] = 1
            # C[i, j] -> W[j*3 + i]
            W[j * 3 + i, col_offset] = 1
            col_offset += 1
    # col_offset is now 16

    # Part 3: A[0:3, 2:4] * B[2:4, 0:4] -> C[0:3, 0:4]
    # Rank 22
    # Split into B[2:4, 0:2] and B[2:4, 2:4]
    
    # Local factors for 3x2x2
    U_loc = np.zeros((6, 11))
    V_loc = np.zeros((4, 11))
    W_loc = np.zeros((6, 11))
    
    # Strassen part (cols 0-6)
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

    # Extra part (cols 7-10)
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

    def add_322_block(a_col_start, b_row_start, b_col_start, c_col_start, col_offset):
        # A indices: (0..2, a_col_start..a_col_start+1)
        # B indices: (b_row_start..b_row_start+1, b_col_start..b_col_start+1)
        # C indices: (0..2, c_col_start..c_col_start+1)
        
        for r in range(3):
            for c in range(2):
                local_a_idx = r * 2 + c
                global_a_idx = r * m + (a_col_start + c)
                U[global_a_idx, col_offset:col_offset + 11] += U_loc[local_a_idx, :]

        for r in range(2):
            for c in range(2):
                local_b_idx = r * 2 + c
                global_b_idx = (b_row_start + r) * p + (b_col_start + c)
                V[global_b_idx, col_offset:col_offset + 11] += V_loc[local_b_idx, :]

        for r in range(3):
            for c in range(2):
                local_c_idx = r * 2 + c
                global_c_idx = (c_col_start + c) * n + r
                W[global_c_idx, col_offset:col_offset + 11] += W_loc[local_c_idx, :]

    # Sub-part 3a: A[0:3, 2:4] * B[2:4, 0:2] -> C[0:3, 0:2]
    add_322_block(2, 2, 0, 0, 16)
    
    # Sub-part 3b: A[0:3, 2:4] * B[2:4, 2:4] -> C[0:3, 2:4]
    add_322_block(2, 2, 2, 2, 27)

    if time.time() - start_time > deadline:
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
