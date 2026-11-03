import time
import numpy as np

DIMENSIONS = [2, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    """
    Returns a rank-28 tensor decomposition for 2x4 * 4x4 matrix multiplication.
    Uses 4 independent Strassen 2x2x2 blocks.
    """
    n, m, p = DIMENSIONS
    rank = 28

    U = [[0] * rank for _ in range(n * m)]  # shape (8, 28)
    V = [[0] * rank for _ in range(m * p)]  # shape (16, 28)
    W = [[0] * rank for _ in range(p * n)]  # shape (8, 28)

    # Strassen 2x2x2 coefficients
    # A indices: 00=0, 01=1, 10=2, 11=3
    # B indices: 00=0, 01=1, 10=2, 11=3
    # C indices: 00=0, 01=1, 10=2, 11=3
    
    # Each step is (u_dict, v_dict, w_dict)
    strassen_steps = [
        ({0: 1, 3: 1}, {0: 1, 3: 1}, {0: 1, 3: 1}),   # M1
        ({2: 1, 3: 1}, {0: 1},       {2: 1, 3: -1}),   # M2
        ({0: 1},       {1: 1, 3: -1},{1: 1, 3: 1}),    # M3
        ({3: 1},       {2: 1, 0: -1},{0: 1, 2: 1}),    # M4
        ({0: 1, 1: 1}, {3: 1},       {0: -1, 1: 1}),   # M5
        ({2: 1, 0: -1},{0: 1, 1: 1}, {3: 1}),          # M6
        ({1: 1, 3: -1},{2: 1, 3: 1}, {0: 1})           # M7
    ]

    # Define the 4 blocks: (a_row, a_col_start, b_row_start, b_col_start, c_col_start)
    # A is 2x4, B is 4x4, C is 2x4
    # Block 1: A[:, 0:2] * B[0:2, 0:2] -> C[:, 0:2]
    # Block 2: A[:, 0:2] * B[0:2, 2:4] -> C[:, 2:4]
    # Block 3: A[:, 2:4] * B[2:4, 0:2] -> C[:, 0:2]
    # Block 4: A[:, 2:4] * B[2:4, 2:4] -> C[:, 2:4]
    
    blocks = [
        (0, 0, 0, 0),
        (0, 0, 0, 2),
        (0, 2, 2, 0),
        (0, 2, 2, 2)
    ]
    
    # Note: a_row is always 0 because A has only 2 rows, and we are splitting columns.
    # The "a_row" in the tuple above is actually the start row of A, which is always 0.
    # The "a_col_start" is the start column of A.
    
    for b_idx, (a_row_start, a_col_start, b_row_start, b_col_start, c_col_start) in enumerate(blocks):
        col_offset = b_idx * 7
        
        for s_idx in range(7):
            u_dict, v_dict, w_dict = strassen_steps[s_idx]
            col = col_offset + s_idx
            
            # Update U
            # U index: i*m + j
            for local_idx, coeff in u_dict.items():
                l_row = local_idx // 2
                l_col = local_idx % 2
                global_i = a_row_start + l_row
                global_j = a_col_start + l_col
                global_idx = global_i * m + global_j
                U[global_idx][col] += coeff
            
            # Update V
            # V index: j*p + k
            for local_idx, coeff in v_dict.items():
                l_row = local_idx // 2
                l_col = local_idx % 2
                global_j = b_row_start + l_row
                global_k = b_col_start + l_col
                global_idx = global_j * p + global_k
                V[global_idx][col] += coeff
            
            # Update W
            # W index: k*n + i
            for local_idx, coeff in w_dict.items():
                l_row = local_idx // 2
                l_col = local_idx % 2
                global_i = l_row
                global_k = c_col_start + l_col
                global_idx = global_k * n + global_i
                W[global_idx][col] += coeff

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
