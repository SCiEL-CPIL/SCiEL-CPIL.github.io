import time
import numpy as np

DIMENSIONS = [2, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Rank 56 decomposition using Strassen for 2x2 blocks
    # Structure:
    # A is 2x4, split into A_L (2x2, cols 0,1) and A_R (2x2, cols 2,3)
    # B is 4x8, split into B_top (2x8, rows 0,1) and B_bot (2x8, rows 2,3)
    # C = A_L @ B_top + A_R @ B_bot
    # Each term is 2x2 by 2x8.
    # 2x2 by 2x8 is 4 independent 2x2 by 2x2 multiplications.
    # Each 2x2 by 2x2 uses Strassen (rank 7).
    # Total rank = 2 (halves) * 4 (col blocks) * 7 (Strassen) = 56
    
    rank = 56
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    # Strassen coefficients for 2x2x2
    # Local indices for A (2x2): 00, 01, 10, 11 -> 0, 1, 2, 3
    # Local indices for B (2x2): 00, 01, 10, 11 -> 0, 1, 2, 3
    # Local indices for C (2x2): 00, 01, 10, 11 -> 0, 1, 2, 3
    
    # M1 = (A00 + A11) * (B00 + B11)
    # M2 = (A10 + A11) * B00
    # M3 = A00 * (B01 - B11)
    # M4 = A11 * (B10 - B00)
    # M5 = (A00 + A01) * B11
    # M6 = (A10 - A00) * (B00 + B01)
    # M7 = (A01 - A11) * (B10 + B11)
    
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # Define Strassen U, V, W for local 2x2x2
    # U rows: A00, A01, A10, A11
    # V rows: B00, B01, B10, B11
    # W rows: C00, C01, C10, C11
    
    U_s = np.zeros((4, 7))
    V_s = np.zeros((4, 7))
    W_s = np.zeros((4, 7))
    
    # Col 0: M1
    U_s[0, 0] = 1; U_s[3, 0] = 1
    V_s[0, 0] = 1; V_s[3, 0] = 1
    W_s[0, 0] = 1; W_s[3, 0] = 1
    
    # Col 1: M2
    U_s[2, 1] = 1; U_s[3, 1] = 1
    V_s[0, 1] = 1
    W_s[2, 1] = 1; W_s[3, 1] = -1
    
    # Col 2: M3
    U_s[0, 2] = 1
    V_s[1, 2] = 1; V_s[3, 2] = -1
    W_s[1, 2] = 1; W_s[3, 2] = 1
    
    # Col 3: M4
    U_s[3, 3] = 1
    V_s[2, 3] = 1; V_s[0, 3] = -1
    W_s[0, 3] = 1; W_s[2, 3] = 1
    
    # Col 4: M5
    U_s[0, 4] = 1; U_s[1, 4] = 1
    V_s[3, 4] = 1
    W_s[0, 4] = -1; W_s[1, 4] = 1
    
    # Col 5: M6
    U_s[2, 5] = 1; U_s[0, 5] = -1
    V_s[0, 5] = 1; V_s[1, 5] = 1
    W_s[3, 5] = 1
    
    # Col 6: M7
    U_s[1, 6] = 1; U_s[3, 6] = -1
    V_s[2, 6] = 1; V_s[3, 6] = 1
    W_s[0, 6] = 1; W_s[3, 6] = 1

    # Global mapping
    # Block 1: A_L (j=0,1), B_top (j=0,1)
    # Block 2: A_R (j=2,3), B_bot (j=2,3)
    
    for half in range(2): # 0: Left/Top, 1: Right/Bot
        a_offset = half * 2 # j offset for A (0 or 2)
        b_offset = half * 2 # j offset for B (0 or 2)
        
        for col_block in range(4): # 4 blocks of 2 columns in p=8
            k_offset = col_block * 2 # k offset (0, 2, 4, 6)
            
            # Map local Strassen indices to global indices
            # Local A indices: 00, 01, 10, 11
            # Global A index: i * m + j
            # i=0,1. j = a_offset, a_offset+1.
            
            # Local B indices: 00, 01, 10, 11
            # Global B index: j * p + k
            # j = b_offset, b_offset+1. k = k_offset, k_offset+1.
            
            # Local C indices: 00, 01, 10, 11
            # Global C index: k * n + i
            # i=0,1. k = k_offset, k_offset+1.
            
            for s_col in range(7):
                global_col = half * 28 + col_block * 7 + s_col
                
                # Fill U
                for local_i in range(2):
                    for local_j in range(2):
                        local_idx = local_i * 2 + local_j
                        val = U_s[local_idx, s_col]
                        if val != 0:
                            global_i = local_i
                            global_j = a_offset + local_j
                            global_u_idx = global_i * m + global_j
                            U[global_u_idx, global_col] = val
                
                # Fill V
                for local_j in range(2):
                    for local_k in range(2):
                        local_idx = local_j * 2 + local_k
                        val = V_s[local_idx, s_col]
                        if val != 0:
                            global_j = b_offset + local_j
                            global_k = k_offset + local_k
                            global_v_idx = global_j * p + global_k
                            V[global_v_idx, global_col] = val
                
                # Fill W
                for local_i in range(2):
                    for local_k in range(2):
                        local_idx = local_i * 2 + local_k
                        val = W_s[local_idx, s_col]
                        if val != 0:
                            global_i = local_i
                            global_k = k_offset + local_k
                            global_w_idx = global_k * n + global_i
                            W[global_w_idx, global_col] = val

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
