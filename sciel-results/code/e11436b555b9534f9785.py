import numpy as np
import time

DIMENSIONS = [4, 4, 7]

def construct():
    start_time = time.time()
    deadline = start_time + 140
    
    n, m, p = DIMENSIONS
    
    # New Rank: 49 (for 4x4x4) + 28 (for 4x4x2) + 16 (for 4x4x1) = 93
    rank = 93
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # --- Part 1: 4x4 * 4x4 (Left part of B, cols 0-3) ---
    # Strassen 4x4x4 (Rank 49)
    
    u_in = [
        [1, 0, 0, 1],
        [0, 0, 1, 1],
        [1, 0, 0, 0],
        [0, 0, 0, 1],
        [1, 1, 0, 0],
        [-1, 0, 1, 0],
        [0, 1, 0, -1]
    ]
    v_in = [
        [1, 0, 0, 1],
        [1, 0, 0, 0],
        [0, 1, 0, -1],
        [-1, 0, 1, 0],
        [0, 0, 0, 1],
        [1, 1, 0, 0],
        [0, 0, 1, 1]
    ]
    w_out = [
        [1, 0, 0, 1, -1, 0, 1],
        [0, 0, 1, 0, 1, 0, 0],
        [0, 1, 0, 1, 0, 0, 0],
        [1, -1, 1, 0, 0, 1, 0]
    ]
    
    u_out = [
        [1, 0, 0, 1],
        [0, 0, 1, 1],
        [1, 0, 0, 0],
        [0, 0, 0, 1],
        [1, 1, 0, 0],
        [-1, 0, 1, 0],
        [0, 1, 0, -1]
    ]
    v_out = [
        [1, 0, 0, 1],
        [1, 0, 0, 0],
        [0, 1, 0, -1],
        [-1, 0, 1, 0],
        [0, 0, 0, 1],
        [1, 1, 0, 0],
        [0, 0, 1, 1]
    ]
    w_outer = [
        [1, 0, 0, 1, -1, 0, 1],
        [0, 0, 1, 0, 1, 0, 0],
        [0, 1, 0, 1, 0, 0, 0],
        [1, -1, 1, 0, 0, 1, 0]
    ]
    
    for k in range(7):
        for l in range(7):
            if time.time() > deadline:
                break
            u_vec = [0] * 16
            for i_inner in range(2):
                for j_inner in range(2):
                    coeff_inner = u_in[l][i_inner * 2 + j_inner]
                    if coeff_inner == 0: continue
                    for b in range(4):
                        coeff_outer = u_out[k][b]
                        if coeff_outer == 0: continue
                        b_row = b // 2
                        b_col = b % 2
                        i_global = 2 * b_row + i_inner
                        j_global = 2 * b_col + j_inner
                        u_vec[i_global * 4 + j_global] += coeff_inner * coeff_outer
            
            v_vec = [0] * 28
            for i_inner in range(2):
                for j_inner in range(2):
                    coeff_inner = v_in[l][i_inner * 2 + j_inner]
                    if coeff_inner == 0: continue
                    for b in range(4):
                        coeff_outer = v_out[k][b]
                        if coeff_outer == 0: continue
                        b_row = b // 2
                        b_col = b % 2
                        j_global = 2 * b_row + i_inner
                        k_global = 2 * b_col + j_inner
                        v_vec[j_global * 7 + k_global] += coeff_inner * coeff_outer
            
            w_vec = [0] * 28
            for c_block in range(4):
                coeff_outer_w = w_outer[c_block][k]
                if coeff_outer_w == 0: continue
                for c_inner in range(4):
                    coeff_inner_w = w_out[c_inner][l]
                    if coeff_inner_w == 0: continue
                    total_coeff = coeff_outer_w * coeff_inner_w
                    if total_coeff == 0: continue
                    
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
        if time.time() > deadline:
            break
    
    # --- Part 2: 4x4 * 4x3 (Right part of B, cols 4-6) ---
    # Split into 4x4x2 (Rank 28) and 4x4x1 (Rank 16)
    
    # Helper for Strassen 2x2x2
    # u_s, v_s, w_s are 2x2 matrices (flattened 4)
    # We reuse the 2x2x2 Strassen coefficients
    u_2x2 = [
        [1, 0, 0, 1],
        [0, 0, 1, 1],
        [1, 0, 0, 0],
        [0, 0, 0, 1],
        [1, 1, 0, 0],
        [-1, 0, 1, 0],
        [0, 1, 0, -1]
    ]
    v_2x2 = [
        [1, 0, 0, 1],
        [1, 0, 0, 0],
        [0, 1, 0, -1],
        [-1, 0, 1, 0],
        [0, 0, 0, 1],
        [1, 1, 0, 0],
        [0, 0, 1, 1]
    ]
    w_2x2 = [
        [1, 0, 0, 1, -1, 0, 1],
        [0, 0, 1, 0, 1, 0, 0],
        [0, 1, 0, 1, 0, 0, 0],
        [1, -1, 1, 0, 0, 1, 0]
    ]

    # Subpart 2a: 4x4 * 4x2 (Cols 4, 5)
    # A blocks: A00, A01, A10, A11
    # B blocks: BL0 (rows 0,1 cols 4,5), BL1 (rows 2,3 cols 4,5)
    # C blocks: CL0 (rows 0,1 cols 4,5), CL1 (rows 2,3 cols 4,5)
    # CL0 = A00*BL0 + A01*BL1
    # CL1 = A10*BL0 + A11*BL1
    
    # Terms:
    # 1. A00 * BL0 -> CL0
    # 2. A01 * BL1 -> CL0
    # 3. A10 * BL0 -> CL1
    # 4. A11 * BL1 -> CL1
    
    block_configs_2x2 = [
        # (A_block_row, A_block_col, B_block_row, B_block_col, C_block_row, C_block_col)
        # A is 4x4, B is 4x2, C is 4x2
        # A blocks are 2x2.
        # B blocks are 2x2.
        # C blocks are 2x2.
        
        # A00 (r=0,c=0) * BL0 (r=0,c=0 of 4x2) -> CL0 (r=0,c=0 of 4x2)
        # BL0 is B rows 0-1, cols 4-5. Global B col offset 4.
        # CL0 is C rows 0-1, cols 4-5. Global C col offset 4.
        (0, 0, 0, 0, 0, 0),
        
        # A01 (r=0,c=1) * BL1 (r=1,c=0 of 4x2) -> CL0 (r=0,c=0 of 4x2)
        # A01 is A rows 0-1, cols 2-3.
        # BL1 is B rows 2-3, cols 4-5.
        (0, 1, 1, 0, 0, 0),
        
        # A10 (r=1,c=0) * BL0 (r=0,c=0 of 4x2) -> CL1 (r=1,c=0 of 4x2)
        # A10 is A rows 2-3, cols 0-1.
        # CL1 is C rows 2-3, cols 4-5.
        (1, 0, 0, 0, 1, 0),
        
        # A11 (r=1,c=1) * BL1 (r=1,c=0 of 4x2) -> CL1 (r=1,c=0 of 4x2)
        # A11 is A rows 2-3, cols 2-3.
        (1, 1, 1, 0, 1, 0),
    ]
    
    for a_r, a_c, b_r, b_c, c_r, c_c in block_configs_2x2:
        for k in range(7):
            for l in range(7):
                if time.time() > deadline:
                    break
                
                # U: A block
                u_vec = [0] * 16
                for i in range(2):
                    for j in range(2):
                        coeff = u_2x2[l][i*2+j]
                        if coeff == 0: continue
                        gi = 2*a_r + i
                        gj = 2*a_c + j
                        u_vec[gi*4 + gj] = coeff
                
                # V: B block
                v_vec = [0] * 28
                for i in range(2):
                    for j in range(2):
                        coeff = v_2x2[l][i*2+j]
                        if coeff == 0: continue
                        gi = 2*b_r + i
                        gj = 4 + 2*b_c + j # B col offset 4
                        v_vec[gi*7 + gj] = coeff
                
                # W: C block
                w_vec = [0] * 28
                for i in range(2):
                    for j in range(2):
                        coeff = w_2x2[i*2+j][k] # Note: w_2x2 is [c_index][k_index] in previous logic?
                        # In previous code: w_out[c_inner][l] where l is inner index.
                        # Here w_2x2 is defined as w_out.
                        # w_2x2[row][col] where row is c_inner, col is k?
                        # Previous code: w_out = [[1, 0, 0, 1, -1, 0, 1], ...]
                        # w_out[c_inner][l]
                        # So w_2x2[i*2+j][k] is correct.
                        if coeff == 0: continue
                        gi = 2*c_r + i
                        gj = 4 + 2*c_c + j # C col offset 4
                        w_vec[gj*4 + gi] = coeff # W index is [k*n+i, r] -> [col, row]?
                        # Task: W[k*n+i, r].
                        # k is col of C, i is row of C.
                        # So index is k*n + i.
                        # n=4.
                        # So W[gj*4 + gi]. Correct.
                
                for r in range(16):
                    U[r][col] = u_vec[r]
                for r in range(28):
                    V[r][col] = v_vec[r]
                for r in range(28):
                    W[r][col] = w_vec[r]
                
                col += 1
            if time.time() > deadline:
                break
        if time.time() > deadline:
            break

    # Subpart 2b: 4x4 * 4x1 (Col 6)
    # A blocks: A00, A01, A10, A11
    # B blocks: b0 (rows 0,1 col 6), b1 (rows 2,3 col 6)
    # C blocks: CR0 (rows 0,1 col 6), CR1 (rows 2,3 col 6)
    # CR0 = A00*b0 + A01*b1
    # CR1 = A10*b0 + A11*b1
    
    block_configs_2x1 = [
        # (A_block_row, A_block_col, B_block_row, C_block_row)
        # B col is always 6.
        # C col is always 6.
        
        # A00 * b0 -> CR0
        (0, 0, 0, 0),
        
        # A01 * b1 -> CR0
        (0, 1, 1, 0),
        
        # A10 * b0 -> CR1
        (1, 0, 0, 1),
        
        # A11 * b1 -> CR1
        (1, 1, 1, 1),
    ]
    
    # Schoolbook 2x2x1
    # U: A (2x2), V: b (2x1), W: c (2x1)
    # 4 terms:
    # (0,0) * (0) -> (0)
    # (0,1) * (1) -> (0)
    # (1,0) * (0) -> (1)
    # (1,1) * (1) -> (1)
    
    schoolbook_2x1 = [
        (0, 0, 0, 0), # A(i,j), B(j), C(i)
        (0, 1, 1, 0),
        (1, 0, 0, 1),
        (1, 1, 1, 1),
    ]
    
    for a_r, a_c, b_r, c_r in block_configs_2x1:
        for si, sj, sb, sc in schoolbook_2x1:
            if time.time() > deadline:
                break
            
            u_vec = [0] * 16
            gi = 2*a_r + si
            gj = 2*a_c + sj
            u_vec[gi*4 + gj] = 1
            
            v_vec = [0] * 28
            gi = 2*b_r + sb
            gj = 6
            v_vec[gi*7 + gj] = 1
            
            w_vec = [0] * 28
            gi = 2*c_r + sc
            gj = 6
            w_vec[gj*4 + gi] = 1
            
            for r in range(16):
                U[r][col] = u_vec[r]
            for r in range(28):
                V[r][col] = v_vec[r]
            for r in range(28):
                W[r][col] = w_vec[r]
            
            col += 1
        if time.time() > deadline:
            break

    return {"U": U, "V": V, "W": W}
