import numpy as np
import time

DIMENSIONS = [4, 4, 7]

def construct():
    """
    Construct a rank-minimal exact tensor decomposition for multiplying
    a 4x4 matrix by a 4x7 matrix.
    
    Interface: construct() -> {'U', 'V', 'W'}
    Shapes: U (n*m, rank), V (m*p, rank), W (p*n, rank)
    Convention: W[k*n+i, r]
    
    Strategy:
    Split the multiplication into three independent parts:
    1. Left part: A (4x4) * B_left (4x4) -> C_left (4x4)
       Use Strassen's algorithm for 4x4x4 with rank 49.
       
    2. Middle part: A (4x4) * B_mid (4x2) -> C_mid (4x2)
       Use 4 Strassen 2x2x2 blocks (2 for top half of C, 2 for bottom half).
       Rank = 4 * 7 = 28.
       
    3. Right part: A (4x4) * B_right (4x1) -> C_right (4x1)
       Use schoolbook multiplication with rank 4*4*1 = 16.
       
    Total rank = 49 + 28 + 16 = 93.
    """
    
    deadline = time.time() + 120
    
    n, m, p = DIMENSIONS  # n=4, m=4, p=7
    
    # Strassen 4x4x4 (Rank 49)
    u_in = [
        [1, 0, 0, 1], # M1
        [0, 0, 1, 1], # M2
        [1, 0, 0, 0], # M3
        [0, 0, 0, 1], # M4
        [1, 1, 0, 0], # M5
        [-1, 0, 1, 0],# M6
        [0, 1, 0, -1] # M7
    ]
    v_in = [
        [1, 0, 0, 1], # M1
        [1, 0, 0, 0], # M2
        [0, 1, 0, -1],# M3
        [-1, 0, 1, 0],# M4
        [0, 0, 0, 1], # M5
        [1, 1, 0, 0], # M6
        [0, 0, 1, 1]  # M7
    ]
    w_out = [
        [1, 0, 0, 1, -1, 0, 1], # C00
        [0, 0, 1, 0, 1, 0, 0],  # C01
        [0, 1, 0, 1, 0, 0, 0],  # C10
        [1, -1, 1, 0, 0, 1, 0]  # C11
    ]
    
    u_out = u_in
    v_out = v_in
    w_outer = w_out
    
    rank_strassen_4x4 = 49
    rank_strassen_4x4x2 = 28
    rank_schoolbook_4x4x1 = 16
    rank = rank_strassen_4x4 + rank_strassen_4x4x2 + rank_schoolbook_4x4x1 # 93
    
    U = [[0] * rank for _ in range(n * m)]  # 16 x 93
    V = [[0] * rank for _ in range(m * p)]  # 28 x 93
    W = [[0] * rank for _ in range(p * n)]  # 28 x 93
    
    col = 0
    
    # --- Part 1: 4x4 * 4x4 (Cols 0-3) ---
    for k_outer in range(7):
        for l_inner in range(7):
            u_vec = [0] * 16
            for i_inner in range(2):
                for j_inner in range(2):
                    coeff_inner = u_in[l_inner][i_inner * 2 + j_inner]
                    if coeff_inner == 0:
                        continue
                    for b in range(4):
                        coeff_outer = u_out[k_outer][b]
                        if coeff_outer == 0:
                            continue
                        b_row = b // 2
                        b_col = b % 2
                        i_global = 2 * b_row + i_inner
                        j_global = 2 * b_col + j_inner
                        u_vec[i_global * 4 + j_global] += coeff_inner * coeff_outer
            
            v_vec = [0] * 28
            for i_inner in range(2):
                for j_inner in range(2):
                    coeff_inner = v_in[l_inner][i_inner * 2 + j_inner]
                    if coeff_inner == 0:
                        continue
                    for b in range(4):
                        coeff_outer = v_out[k_outer][b]
                        if coeff_outer == 0:
                            continue
                        b_row = b // 2
                        b_col = b % 2
                        j_global = 2 * b_row + i_inner
                        k_global = 2 * b_col + j_inner
                        v_vec[j_global * 7 + k_global] += coeff_inner * coeff_outer
            
            w_vec = [0] * 28
            for c_block in range(4):
                coeff_outer_w = w_outer[c_block][k_outer]
                if coeff_outer_w == 0:
                    continue
                for c_inner in range(4):
                    coeff_inner_w = w_out[c_inner][l_inner]
                    if coeff_inner_w == 0:
                        continue
                    total_coeff = coeff_outer_w * coeff_inner_w
                    if total_coeff == 0:
                        continue
                    c_block_row = c_block // 2
                    c_block_col = c_block % 2
                    c_inner_row = c_inner // 2
                    c_inner_col = c_inner % 2
                    i_global = 2 * c_block_row + c_inner_row
                    k_global = 2 * c_block_col + c_inner_col
                    w_vec[k_global * 4 + i_global] += total_coeff
            
            for r in range(16):
                U[r][col] = u_vec[r]
            for r in range(28):
                V[r][col] = v_vec[r]
            for r in range(28):
                W[r][col] = w_vec[r]
            
            col += 1
            
            if time.time() > deadline - 5:
                break
        if time.time() > deadline - 5:
            break

    if col >= rank_strassen_4x4 and time.time() < deadline - 5:
        # --- Part 2: 4x4 * 4x2 (Cols 4-5) ---
        # Uses 4 Strassen 2x2x2 blocks.
        # C_top = A_00 B_top + A_01 B_bot
        # C_bot = A_10 B_top + A_11 B_bot
        # B is 4x2. B_top = B[0:2, 4:6], B_bot = B[2:4, 4:6]
        # A is 4x4. A_00 = A[0:2, 0:2], A_01 = A[0:2, 2:4]
        #          A_10 = A[2:4, 0:2], A_11 = A[2:4, 2:4]
        
        # Define the 4 blocks
        # Block 0: A_00 * B_top
        # Block 1: A_01 * B_bot
        # Block 2: A_10 * B_top
        # Block 3: A_11 * B_bot
        
        block_defs = [
            # (A_row_offset, A_col_offset, B_row_offset, B_col_offset, C_row_offset, C_col_offset)
            (0, 0, 0, 4, 0, 4), # A_00 * B_top -> C_top_left
            (0, 2, 2, 4, 0, 4), # A_01 * B_bot -> C_top_right (adds to C_top)
            (2, 0, 0, 4, 2, 4), # A_10 * B_top -> C_bot_left
            (2, 2, 2, 4, 2, 4), # A_11 * B_bot -> C_bot_right (adds to C_bot)
        ]
        
        for bd in block_defs:
            a_ro, a_co, b_ro, b_co, c_ro, c_co = bd
            
            for k_inner in range(7):
                u_vec = [0] * 16
                v_vec = [0] * 28
                w_vec = [0] * 28
                
                # U: A indices
                for i_inner in range(2):
                    for j_inner in range(2):
                        coeff_inner = u_in[k_inner][i_inner * 2 + j_inner]
                        if coeff_inner == 0:
                            continue
                        i_global = a_ro + i_inner
                        j_global = a_co + j_inner
                        u_vec[i_global * 4 + j_global] = coeff_inner
                
                # V: B indices
                # B is 4x7. We are using cols 4,5.
                # B_top is rows 0,1 cols 4,5.
                # B_bot is rows 2,3 cols 4,5.
                for i_inner in range(2):
                    for j_inner in range(2):
                        coeff_inner = v_in[k_inner][i_inner * 2 + j_inner]
                        if coeff_inner == 0:
                            continue
                        j_global = b_ro + i_inner
                        k_global = b_co + j_inner
                        v_vec[j_global * 7 + k_global] = coeff_inner
                
                # W: C indices
                # C is 4x7. We are filling cols 4,5.
                # C_top is rows 0,1 cols 4,5.
                # C_bot is rows 2,3 cols 4,5.
                for c_inner in range(4):
                    coeff_inner_w = w_out[c_inner][k_inner]
                    if coeff_inner_w == 0:
                        continue
                    c_inner_row = c_inner // 2
                    c_inner_col = c_inner % 2
                    i_global = c_ro + c_inner_row
                    k_global = c_co + c_inner_col
                    w_vec[k_global * 4 + i_global] += coeff_inner_w
                
                for r in range(16):
                    U[r][col] = u_vec[r]
                for r in range(28):
                    V[r][col] = v_vec[r]
                for r in range(28):
                    W[r][col] = w_vec[r]
                
                col += 1
                
                if time.time() > deadline - 5:
                    break
            if time.time() > deadline - 5:
                break

    if col >= rank_strassen_4x4 + rank_strassen_4x4x2 and time.time() < deadline - 5:
        # --- Part 3: 4x4 * 4x1 (Col 6) ---
        # Schoolbook Rank 16
        for i in range(4):
            for j in range(4):
                U[i * 4 + j][col] = 1
                V[j * 7 + 6][col] = 1
                W[6 * 4 + i][col] = 1
                col += 1

    return {"U": U, "V": V, "W": W}

BET: {mechanism: "The current program already implements the optimal split (49+28+16=93). The inspirations offer no better rank. The only improvement is ensuring the code is robust and fast enough to complete within the deadline, which it already does. The score should remain at 1.2043.", q: 0.5, delta: [0.0, 0.0]}
