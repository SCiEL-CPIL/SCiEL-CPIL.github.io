import time

DIMENSIONS = [4, 4, 5]

def construct():
    n, m, p = DIMENSIONS
    # We will build the decomposition in parts
    # Part 1: 4x4 * 4x4 (first 4 columns of B) using Strassen (Rank 49)
    # Part 2: 4x4 * 4x1 (last column of B) using Schoolbook (Rank 16)
    
    U_parts = []
    V_parts = []
    W_parts = []
    
    # --- Part 1: Strassen for 4x4 * 4x4 ---
    # Strassen 2x2 base case terms (u, v, w)
    # u, v, w are lists of length 4
    # Indices for 2x2:
    # A: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
    # B: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
    # C: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
    
    s2x2_terms = [
        # M1
        ([1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, 1]),
        # M2
        ([0, 0, 1, 1], [1, 0, 0, 0], [0, 0, 1, 0]),
        # M3
        ([1, 0, 0, 0], [0, 1, 0, -1], [0, 1, 0, 1]),
        # M4
        ([0, 0, 0, 1], [0, 0, 1, -1], [-1, 1, 0, 0]),
        # M5
        ([1, 1, 0, 0], [0, 0, 0, 1], [1, 0, 1, 0]),
        # M6
        ([-1, 0, 1, 0], [1, 1, 0, 0], [0, 0, 0, 1]),
        # M7
        ([0, 1, -1, 0], [0, 0, 1, 1], [-1, 0, 0, 0]),
    ]
    
    # Strassen 4x4 structure
    # P1 = (A00+A11)(B00+B11)
    # P2 = (A10+A11)B00
    # P3 = A00(B01-B11)
    # P4 = A11(B10-B00)
    # P5 = (A00+A01)B11
    # P6 = (A10-A00)(B00+B01)
    # P7 = (A01-A11)(B10+B11)
    
    # Coefficients for A blocks in Pk
    # A blocks: 00, 01, 10, 11
    # Pk uses A_linear_comb
    A_coeffs = {
        1: {0: 1, 1: 0, 2: 0, 3: 1}, # A00 + A11
        2: {0: 0, 1: 0, 2: 1, 3: 1}, # A10 + A11
        3: {0: 1, 1: 0, 2: 0, 3: 0}, # A00
        4: {0: 0, 1: 0, 2: 0, 3: 1}, # A11
        5: {0: 1, 1: 1, 2: 0, 3: 0}, # A00 + A01
        6: {0: -1, 1: 0, 2: 1, 3: 0}, # A10 - A00
        7: {0: 0, 1: 1, 2: 0, 3: -1}, # A01 - A11
    }
    
    B_coeffs = {
        1: {0: 1, 1: 0, 2: 0, 3: 1}, # B00 + B11
        2: {0: 1, 1: 0, 2: 0, 3: 0}, # B00
        3: {0: 0, 1: 1, 2: 0, 3: -1}, # B01 - B11
        4: {0: -1, 1: 0, 2: 1, 3: 0}, # B10 - B00
        5: {0: 0, 1: 0, 2: 0, 3: 1}, # B11
        6: {0: 1, 1: 1, 2: 0, 3: 0}, # B00 + B01
        7: {0: 0, 1: 0, 2: 1, 3: 1}, # B10 + B11
    }
    
    # Coefficients for C blocks from Pk
    # C blocks: 00, 01, 10, 11
    # C00 = P1 - P4 + P5 - P7
    # C01 = P3 + P4
    # C10 = P2 + P5
    # C11 = P1 - P2 + P3 + P6
    C_coeffs = {
        1: {0: 1, 1: 0, 2: 0, 3: 1}, # P1 contributes to C00 (1) and C11 (1)
        2: {0: 0, 1: 0, 2: 1, 3: -1}, # P2 contributes to C10 (1) and C11 (-1)
        3: {0: 0, 1: 1, 2: 0, 3: 1}, # P3 contributes to C01 (1) and C11 (1)
        4: {0: -1, 1: 1, 2: 0, 3: 0}, # P4 contributes to C00 (-1) and C01 (1)
        5: {0: 1, 1: 0, 2: 1, 3: 0}, # P5 contributes to C00 (1) and C10 (1)
        6: {0: 0, 1: 0, 2: 0, 3: 1}, # P6 contributes to C11 (1)
        7: {0: -1, 1: 0, 2: 0, 3: 0}, # P7 contributes to C00 (-1)
    }
    
    # Generate terms for 4x4 * 4x4
    # Global A indices: i*4+j
    # Global B indices: j*4+k
    # Global C indices: k*4+i
    
    for p_idx in range(1, 8):
        a_coeff = A_coeffs[p_idx]
        b_coeff = B_coeffs[p_idx]
        c_coeff = C_coeffs[p_idx]
        
        # For each of the 7 base terms in 2x2 Strassen
        for u_base, v_base, w_base in s2x2_terms:
            # u_base corresponds to A_block (2x2)
            # v_base corresponds to B_block (2x2)
            # w_base corresponds to C_block (2x2)
            
            u_global = [0] * 16
            v_global = [0] * 16
            w_global = [0] * 16
            
            # Map A_base indices to A_global
            # A_base index 0 -> A00(0,0) -> Global 0
            # A_base index 1 -> A00(0,1) -> Global 1
            # A_base index 2 -> A00(1,0) -> Global 4
            # A_base index 3 -> A00(1,1) -> Global 5
            # This mapping depends on which block A_block is.
            # A_block 0 is A00 (rows 0-1, cols 0-1)
            # A_block 1 is A01 (rows 0-1, cols 2-3)
            # A_block 2 is A10 (rows 2-3, cols 0-1)
            # A_block 3 is A11 (rows 2-3, cols 2-3)
            
            for base_idx, val in enumerate(u_base):
                if val == 0: continue
                # base_idx 0,1,2,3 correspond to local (0,0), (0,1), (1,0), (1,1)
                local_i = base_idx // 2
                local_j = base_idx % 2
                
                # We need to sum contributions from all A_blocks that have non-zero coeff in a_coeff
                # Wait, the term is (sum c_b A_b) * (sum d_c B_c) * ...
                # The u_vector for the global term is the linear combination of A_blocks.
                # u_global = sum_{b} a_coeff[b] * u_base_mapped_to_block_b
                
                # So we iterate through blocks b
                for b_idx in range(4):
                    c_a = a_coeff[b_idx]
                    if c_a == 0: continue
                    # Map local (local_i, local_j) to global indices for block b_idx
                    # b_idx 0: row 0-1, col 0-1
                    # b_idx 1: row 0-1, col 2-3
                    # b_idx 2: row 2-3, col 0-1
                    # b_idx 3: row 2-3, col 2-3
                    row_start = (b_idx // 2) * 2
                    col_start = (b_idx % 2) * 2
                    g_i = row_start + local_i
                    g_j = col_start + local_j
                    g_idx = g_i * 4 + g_j
                    u_global[g_idx] += c_a * val
            
            # Map B_base indices to B_global
            # B_block 0 is B00 (rows 0-1, cols 0-1)
            # B_block 1 is B01 (rows 0-1, cols 2-3)
            # B_block 2 is B10 (rows 2-3, cols 0-1)
            # B_block 3 is B11 (rows 2-3, cols 2-3)
            for base_idx, val in enumerate(v_base):
                if val == 0: continue
                local_i = base_idx // 2
                local_j = base_idx % 2
                for b_idx in range(4):
                    c_b = b_coeff[b_idx]
                    if c_b == 0: continue
                    row_start = (b_idx // 2) * 2
                    col_start = (b_idx % 2) * 2
                    g_i = row_start + local_i
                    g_j = col_start + local_j
                    g_idx = g_i * 4 + g_j
                    v_global[g_idx] += c_b * val
            
            # Map C_base indices to C_global
            # C_block 0 is C00 (rows 0-1, cols 0-1)
            # C_block 1 is C01 (rows 0-1, cols 2-3)
            # C_block 2 is C10 (rows 2-3, cols 0-1)
            # C_block 3 is C11 (rows 2-3, cols 2-3)
            for base_idx, val in enumerate(w_base):
                if val == 0: continue
                local_i = base_idx // 2
                local_j = base_idx % 2
                for c_blk in range(4):
                    c_c = c_coeff[c_blk]
                    if c_c == 0: continue
                    row_start = (c_blk // 2) * 2
                    col_start = (c_blk % 2) * 2
                    g_i = row_start + local_i
                    g_j = col_start + local_j
                    # W index is k*n + i = col*4 + row
                    g_idx = g_j * 4 + g_i
                    w_global[g_idx] += c_c * val
            
            U_parts.append(u_global)
            V_parts.append(v_global)
            W_parts.append(w_global)
            
    # --- Part 2: Schoolbook for 4x4 * 4x1 ---
    # A is 4x4, B is 4x1 (column 5 of B, index 4)
    # C is 4x1 (column 5 of C, index 4)
    # Rank 16
    # C_i = sum_j A_ij B_j
    # Term for each (i, j): A_ij * B_j * C_i
    # U index: i*4 + j
    # V index: j*4 + 4 (since B col is 4)
    # W index: 4*4 + i (since C col is 4)
    
    for i in range(4):
        for j in range(4):
            u = [0] * 16
            v = [0] * 20 # m*p = 4*5 = 20
            w = [0] * 20 # p*n = 5*4 = 20
            
            u[i * 4 + j] = 1
            v[j * 5 + 4] = 1 # B index j*5 + 4
            w[4 * 4 + i] = 1 # C index 4*4 + i
            
            U_parts.append(u)
            V_parts.append(v)
            W_parts.append(w)
            
    # Combine
    rank = len(U_parts)
    U = [[0] * rank for _ in range(16)]
    V = [[0] * rank for _ in range(20)]
    W = [[0] * rank for _ in range(20)]
    
    for r in range(rank):
        for i in range(16):
            U[i][r] = U_parts[r][i]
        for i in range(20):
            V[i][r] = V_parts[r][i]
        for i in range(20):
            W[i][r] = W_parts[r][i]
            
    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
