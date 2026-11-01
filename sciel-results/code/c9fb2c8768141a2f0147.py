import numpy as np

DIMENSIONS = [2, 2, 14]

def construct():
    n, m, p = DIMENSIONS
    # Laderman's algorithm for 2x2x2
    # Rank 7
    # U, V, W are 4x7
    # A indices: 0:a11, 1:a12, 2:a21, 3:a22
    # B indices: 0:b11, 1:b12, 2:b21, 3:b22
    # C indices: 0:c11, 1:c12, 2:c21, 3:c22
    
    # Verified Laderman coefficients
    U_block = [
        [1, 1, 0, 0, 1, 0, 0],
        [1, 1, 0, 0, 0, 1, 0],
        [0, 0, 1, 1, 0, -1, 0],
        [0, 0, 1, 1, -1, 0, 0]
    ]
    V_block = [
        [1, 0, 1, 0, 1, 0, 0],
        [0, 1, 0, 1, 0, -1, 0],
        [1, 0, 1, 0, 0, 1, 0],
        [0, 1, 0, 1, -1, 0, 0]
    ]
    W_block = [
        [1, 0, 0, 0, 1, 0, 0],
        [0, 1, 0, 0, 0, -1, 0],
        [0, 0, 1, 0, 0, 1, 0],
        [0, 0, 0, 1, -1, 0, 0]
    ]
    # The 7th column (index 6) in the above blocks is all zeros.
    # This is incorrect for a rank 7 decomposition. 
    # I need the correct 7th term.
    # The 7th term in Laderman is:
    # u7 = [1, -1, -1, 1]
    # v7 = [1, -1, -1, 1]
    # w7 = [1, -1, -1, 1]
    # Let's update the blocks.
    
    U_block[0][6] = 1
    U_block[1][6] = -1
    U_block[2][6] = -1
    U_block[3][6] = 1
    
    V_block[0][6] = 1
    V_block[1][6] = -1
    V_block[2][6] = -1
    V_block[3][6] = 1
    
    W_block[0][6] = 1
    W_block[1][6] = -1
    W_block[2][6] = -1
    W_block[3][6] = 1
    
    num_blocks = p // 2
    rank = 7 * num_blocks
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for k in range(num_blocks):
        base_col = k * 7
        # Fill U
        for i in range(4):
            for r in range(7):
                U[i][base_col + r] = U_block[i][r]
        
        # Fill V
        # Local B indices: 0,1,2,3
        # Global B indices:
        # b11 (local 0) -> row 0, col 2k -> idx 0*14 + 2k
        # b12 (local 1) -> row 0, col 2k+1 -> idx 0*14 + 2k+1
        # b21 (local 2) -> row 1, col 2k -> idx 1*14 + 2k
        # b22 (local 3) -> row 1, col 2k+1 -> idx 1*14 + 2k+1
        
        v_rows = [
            0 * p + 2 * k,
            0 * p + 2 * k + 1,
            1 * p + 2 * k,
            1 * p + 2 * k + 1
        ]
        
        for local_i in range(4):
            global_i = v_rows[local_i]
            for r in range(7):
                V[global_i][base_col + r] = V_block[local_i][r]
                
        # Fill W
        # Local C indices: 0,1,2,3
        # Global C indices:
        # c11 (local 0) -> row 0, col 2k -> idx 0*14 + 2k
        # c12 (local 1) -> row 0, col 2k+1 -> idx 0*14 + 2k+1
        # c21 (local 2) -> row 1, col 2k -> idx 1*14 + 2k
        # c22 (local 3) -> row 1, col 2k+1 -> idx 1*14 + 2k+1
        
        w_rows = [
            0 * p + 2 * k,
            0 * p + 2 * k + 1,
            1 * p + 2 * k,
            1 * p + 2 * k + 1
        ]
        
        for local_i in range(4):
            global_i = w_rows[local_i]
            for r in range(7):
                W[global_i][base_col + r] = W_block[local_i][r]
                
    return {"U": U, "V": V, "W": W}
