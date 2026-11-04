import numpy as np
import time

def construct():
    n, m, p = 4, 4, 8
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s limit
    
    # Rank 98 Strassen-based decomposition
    rank = 98
    
    # Initialize U, V, W
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    # Strassen 2x2x2 decomposition details
    # U_2, V_2, W_2 are 4x7
    U_2 = np.zeros((4, 7))
    V_2 = np.zeros((4, 7))
    W_2 = np.zeros((4, 7))
    
    # c=0
    U_2[0, 0] = 1; U_2[3, 0] = 1
    V_2[0, 0] = 1; V_2[3, 0] = 1
    W_2[0, 0] = 1; W_2[3, 0] = 1
    
    # c=1
    U_2[2, 1] = 1; U_2[3, 1] = 1
    V_2[0, 1] = 1
    W_2[2, 1] = 1; W_2[3, 1] = -1
    
    # c=2
    U_2[0, 2] = 1
    V_2[1, 2] = 1; V_2[3, 2] = -1
    W_2[1, 2] = 1; W_2[3, 2] = 1
    
    # c=3
    U_2[3, 3] = 1
    V_2[2, 3] = 1; V_2[0, 3] = -1
    W_2[0, 3] = 1; W_2[2, 3] = 1
    
    # c=4
    U_2[0, 4] = 1; U_2[1, 4] = 1
    V_2[3, 4] = 1
    W_2[0, 4] = -1; W_2[1, 4] = 1
    
    # c=5
    U_2[2, 5] = 1; U_2[0, 5] = -1
    V_2[0, 5] = 1; V_2[1, 5] = 1
    W_2[3, 5] = 1
    
    # c=6
    U_2[1, 6] = 1; U_2[3, 6] = -1
    V_2[2, 6] = 1; V_2[3, 6] = 1
    W_2[0, 6] = 1
    
    # Block Strassen coefficients for 4x4x4
    # P_k = sum_{blocks} coeff * A_block * B_block
    # C_block = sum_{P_k} coeff * P_k
    
    # A_specs: How to form A_eff for each P_k
    # A_eff is 2x2. Block indices: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
    A_specs = [
        [(1, 0), (1, 3)],   # P0: A00 + A11
        [(1, 2), (1, 3)],   # P1: A10 + A11
        [(1, 0)],           # P2: A00
        [(1, 3)],           # P3: A11
        [(1, 0), (1, 1)],   # P4: A00 + A01
        [(1, 2), (-1, 0)],  # P5: A10 - A00
        [(1, 1), (-1, 3)],  # P6: A01 - A11
    ]
    
    # B_specs: How to form B_eff for each P_k
    B_specs = [
        [(1, 0), (1, 3)],   # P0: B00 + B11
        [(1, 0)],           # P1: B00
        [(1, 1), (-1, 3)],  # P2: B01 - B11
        [(1, 2), (-1, 0)],  # P3: B10 - B00
        [(1, 3)],           # P4: B11
        [(1, 0), (1, 1)],   # P5: B00 + B01
        [(1, 2), (1, 3)],   # P6: B10 + B11
    ]
    
    # C_specs: How P_k contributes to C blocks
    # C00 = P0 + P3 - P4 + P6
    # C01 = P2 + P4
    # C10 = P1 + P3
    # C11 = P0 - P1 + P2 + P5
    C_specs = [
        [(1, 0), (1, 3)],   # P0: C00 + C11
        [(1, 2), (-1, 3)],  # P1: C10 - C11
        [(1, 1), (1, 3)],   # P2: C01 + C11
        [(1, 0), (1, 2)],   # P3: C00 + C10
        [(-1, 0), (1, 1)],  # P4: -C00 + C01
        [(1, 3)],           # P5: C11
        [(1, 0)],           # P6: C00
    ]
    
    # Precompute non-zero entries for U_2, V_2, W_2
    # Each is a list of (row, col, value)
    U2_nz = []
    V2_nz = []
    W2_nz = []
    for r in range(4):
        for c in range(7):
            if U_2[r, c] != 0:
                U2_nz.append((r, c, U_2[r, c]))
            if V_2[r, c] != 0:
                V2_nz.append((r, c, V_2[r, c]))
            if W_2[r, c] != 0:
                W2_nz.append((r, c, W_2[r, c]))
    
    # Precompute block row/col indices for 2x2 blocks
    block_rc = []
    for bi in range(2):
        for bj in range(2):
            for r in range(2):
                for c in range(2):
                    global_r = bi * 2 + r
                    global_c = bj * 2 + c
                    a_eff_idx = r * 2 + c
                    block_rc.append((global_r, global_c, a_eff_idx))
    
    # Now, for each half (L, R) and each P_k (0..6) and each S_m (0..6):
    for half in range(2): # 0 for L, 1 for R
        for k_p in range(7): # P_k
            for m_s in range(7): # S_m
                col = half * 49 + k_p * 7 + m_s
                
                # --- U (Input A) ---
                for coeff_a, block_idx in A_specs[k_p]:
                    bi, bj = divmod(block_idx, 2)
                    for global_r, global_c, a_eff_idx in block_rc:
                        if (global_r // 2 == bi and global_c // 2 == bj):
                            global_a_idx = global_r * 4 + global_c
                            # Find U_2[a_eff_idx, m_s]
                            u_val = U_2[a_eff_idx, m_s]
                            if u_val != 0:
                                U[global_a_idx, col] += coeff_a * u_val
                
                # --- V (Input B) ---
                for coeff_b, block_idx in B_specs[k_p]:
                    bi, bj = divmod(block_idx, 2)
                    for global_r, global_c, b_eff_idx in block_rc:
                        if (global_r // 2 == bi and global_c // 2 == bj):
                            global_k = global_c + half * 4
                            global_b_idx = global_r * 8 + global_k
                            v_val = V_2[b_eff_idx, m_s]
                            if v_val != 0:
                                V[global_b_idx, col] += coeff_b * v_val
                
                # --- W (Output C) ---
                for coeff_c, c_block_idx in C_specs[k_p]:
                    bi, bj = divmod(c_block_idx, 2)
                    for global_r, global_c, p_eff_idx in block_rc:
                        if (global_r // 2 == bi and global_c // 2 == bj):
                            global_i = global_r
                            global_k = global_c + half * 4
                            global_w_idx = global_k * 4 + global_i
                            w_val = W_2[p_eff_idx, m_s]
                            if w_val != 0:
                                W[global_w_idx, col] += coeff_c * w_val
                
                # Check time
                if time.time() > deadline:
                    break
            if time.time() > deadline:
                break
        if time.time() > deadline:
            break
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
