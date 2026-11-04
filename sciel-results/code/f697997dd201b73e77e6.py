import numpy as np
import time

def construct():
    """
    Find a rank-minimal exact tensor decomposition for multiplying a 2 by 4 matrix by a 4 by 8 matrix.
    Uses a block-based Strassen decomposition.
    Rank: 56 (8 blocks * 7 Strassen multiplications).
    """
    n, m, p = 2, 4, 8
    rank = 56
    
    U = np.zeros((n * m, rank), dtype=int)
    V = np.zeros((m * p, rank), dtype=int)
    W = np.zeros((p * n, rank), dtype=int)
    
    # Strassen coefficients for 2x2 multiplication
    # Indices: 0:00, 1:01, 2:10, 3:11
    # Term 1: (a00+a11)(b00+b11) -> C00, C11
    # Term 2: (a10+a11)b00 -> C10, C11
    # Term 3: a00(b01-b11) -> C00, C01
    # Term 4: a11(b10-b00) -> C00, C10
    # Term 5: (a00+a01)b11 -> C00, C01
    # Term 6: (a10-a00)(b00+b01) -> C11
    # Term 7: (a01-a11)(b10+b11) -> C00, C11
    
    # U vectors (coeffs for a00, a01, a10, a11)
    # V vectors (coeffs for b00, b01, b10, b11)
    # W vectors (coeffs for c00, c01, c10, c11)
    
    strassen_U = [
        [1, 0, 0, 1],  # T1
        [0, 0, 1, 1],  # T2
        [1, 0, 0, 0],  # T3
        [0, 0, 0, 1],  # T4
        [1, 1, 0, 0],  # T5
        [-1, 0, 1, 0], # T6
        [0, 1, 0, -1]  # T7
    ]
    
    strassen_V = [
        [1, 0, 0, 1],  # T1
        [1, 0, 0, 0],  # T2
        [0, 1, 0, -1], # T3
        [0, 0, 1, -1], # T4
        [0, 0, 0, 1],  # T5
        [1, 1, 0, 0],  # T6
        [0, 0, 1, 1]   # T7
    ]
    
    strassen_W = [
        [1, 0, 0, 1],  # T1 -> C00, C11
        [0, 0, 1, -1], # T2 -> C10, C11
        [1, 1, 0, 0],  # T3 -> C00, C01
        [-1, 0, 1, 0], # T4 -> C00, C10
        [-1, 1, 0, 0], # T5 -> C00, C01
        [0, 0, 0, 1],  # T6 -> C11
        [1, 0, 0, 1]   # T7 -> C00, C11
    ]
    
    # Define blocks
    # Block b:
    # A block: rows 0-1, cols [a_col_start, a_col_start+2)
    # B block: rows [b_row_start, b_row_start+2), cols [b_col_start, b_col_start+2)
    # C block: rows 0-1, cols [c_col_start, c_col_start+2)
    
    blocks = [
        # (a_col_start, b_row_start, b_col_start)
        (0, 0, 0), # Block 0: A_L B_T1
        (0, 0, 2), # Block 1: A_L B_T2
        (0, 0, 4), # Block 2: A_L B_T3
        (0, 0, 6), # Block 3: A_L B_T4
        (2, 2, 0), # Block 4: A_R B_B1
        (2, 2, 2), # Block 5: A_R B_B2
        (2, 2, 4), # Block 6: A_R B_B3
        (2, 2, 6)  # Block 7: A_R B_B4
    ]
    
    for b_idx, (a_cs, b_rs, b_cs) in enumerate(blocks):
        for s in range(7):
            col = b_idx * 7 + s
            
            # Fill U
            # Local a indices: 0:00, 1:01, 2:10, 3:11
            # Global i in [0,2), j in [a_cs, a_cs+2)
            for li in range(2):
                for lj in range(2):
                    local_idx = li * 2 + lj
                    global_i = li
                    global_j = a_cs + lj
                    global_u_idx = global_i * m + global_j
                    U[global_u_idx, col] = strassen_U[s][local_idx]
            
            # Fill V
            # Local b indices: 0:00, 1:01, 2:10, 3:11
            # Global j in [b_rs, b_rs+2), k in [b_cs, b_cs+2)
            for lj in range(2):
                for lk in range(2):
                    local_idx = lj * 2 + lk
                    global_j = b_rs + lj
                    global_k = b_cs + lk
                    global_v_idx = global_j * p + global_k
                    V[global_v_idx, col] = strassen_V[s][local_idx]
            
            # Fill W
            # Local c indices: 0:00, 1:01, 2:10, 3:11
            # Global i in [0,2), k in [b_cs, b_cs+2)
            # Note: C block cols correspond to b_cs
            for li in range(2):
                for lk in range(2):
                    local_idx = li * 2 + lk
                    global_i = li
                    global_k = b_cs + lk
                    # W index is k * n + i
                    global_w_idx = global_k * n + global_i
                    W[global_w_idx, col] = strassen_W[s][local_idx]
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
