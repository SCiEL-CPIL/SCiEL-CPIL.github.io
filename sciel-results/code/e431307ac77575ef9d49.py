import numpy as np
import time

def construct():
    n, m, p = 4, 4, 6
    total_rank = 77
    U = [[0]*total_rank for _ in range(n*m)]
    V = [[0]*total_rank for _ in range(m*p)]
    W = [[0]*total_rank for _ in range(p*n)]
    
    rank_offset = 0
    
    # Define Strassen 2x2 factors
    # u, v, w are dicts mapping local indices (0..3) to coefficients
    # Local index 0: (0,0), 1: (0,1), 2: (1,0), 3: (1,1)
    
    def get_strassen_factors():
        # Returns list of 7 tuples (u_dict, v_dict, w_dict)
        # u_dict, v_dict, w_dict are lists of 4 coeffs
        # Coeffs can be -1, 0, 1
        factors = []
        
        # P1: (x00+x11)(y00+y11) -> z00(+1), z01(0), z10(0), z11(+1) ?? No.
        # Reconstruction:
        # z00 = P1 - P4 + P5 - P7
        # z01 = P3 + P4
        # z10 = P2 + P3
        # z11 = P1 - P2 + P6 - P3
        
        # We define the 7 products P1..P7.
        # For each Pk, we define the input coeffs for X and Y, and the output coeffs for Z.
        
        # P1
        u1 = [1, 0, 0, 1] # x00+x11
        v1 = [1, 0, 0, 1] # y00+y11
        # Contribution to Z:
        # z00 += 1*P1
        # z11 += 1*P1
        w1 = [1, 0, 0, 1]
        factors.append((u1, v1, w1))
        
        # P2
        u2 = [0, 0, 1, 1] # x10+x11
        v2 = [1, 0, 0, 0] # y00
        # z10 += 1*P2
        # z11 += -1*P2
        w2 = [0, 0, 1, -1]
        factors.append((u2, v2, w2))
        
        # P3
        u3 = [1, 0, 0, 0] # x00
        v3 = [0, 0, 1, -1] # y10-y11
        # z01 += 1*P3
        # z10 += 1*P3
        # z11 += -1*P3
        w3 = [0, 1, 1, -1]
        factors.append((u3, v3, w3))
        
        # P4
        u4 = [0, 0, 0, 1] # x11
        v4 = [0, 1, -1, 0] # y01-y00
        # z00 += -1*P4
        # z01 += 1*P4
        w4 = [-1, 1, 0, 0]
        factors.append((u4, v4, w4))
        
        # P5
        u5 = [1, 1, 0, 0] # x00+x01
        v5 = [0, 0, 0, 1] # y11
        # z00 += 1*P5
        w5 = [1, 0, 0, 0]
        factors.append((u5, v5, w5))
        
        # P6
        u6 = [0, 0, 1, -1] # x10-x00
        v6 = [1, 1, 0, 0] # y00+y01
        # z11 += 1*P6
        w6 = [0, 0, 0, 1]
        factors.append((u6, v6, w6))
        
        # P7
        u7 = [0, 1, 0, -1] # x01-x11
        v7 = [0, 0, 1, 1] # y10+y11
        # z00 += -1*P7
        w7 = [-1, 0, 0, 0]
        factors.append((u7, v7, w7))
        
        return factors

    strassen_factors = get_strassen_factors()
    
    def add_2x2_strassen(a_rows, a_cols, b_rows, b_cols, c_rows, c_cols, offset):
        # a_rows, a_cols: start row/col of A block (size 2x2)
        # b_rows, b_cols: start row/col of B block
        # c_rows, c_cols: start row/col of C block
        
        for r in range(7):
            u_coeffs, v_coeffs, w_coeffs = strassen_factors[r]
            
            # u_coeffs maps local (lr, lc) to coeff
            # Global A index: (a_rows + lr, a_cols + lc)
            # U row index: (a_rows + lr)*m + (a_cols + lc)
            
            # v_coeffs maps local (lr, lc) to coeff
            # Global B index: (b_rows + lr, b_cols + lc)
            # V row index: (b_rows + lr)*p + (b_cols + lc)
            
            # w_coeffs maps local (lr, lc) to coeff
            # Global C index: (c_rows + lr, c_cols + lc)
            # W row index: (c_cols + lc)*n + (c_rows + lr)  # Note: W is [k*n + i, r]
            # k is col of C, i is row of C.
            # C[k, i] = B[j, k] A[i, j]? No.
            # Task: A (n x m) * B (m x p) = C (n x p).
            # My previous analysis:
            # U: (n*m, rank). Index i*m+j.
            # V: (m*p, rank). Index j*p+k.
            # W: (p*n, rank). Index k*n+i.
            
            # So W row index for C element (c_row, c_col) is:
            # k = c_col, i = c_row.
            # Index = c_col * n + c_row.
            
            # Wait, let's verify the W index convention.
            # "W[k*n+i,r]"
            # k is from B (m x p) col index? No, B is m x p. j is 0..m-1, k is 0..p-1.
            # A is n x m. i is 0..n-1, j is 0..m-1.
            # C is n x p. i is 0..n-1, k is 0..p-1.
            # So W index is indeed k*n + i.
            
            for lr in range(2):
                for lc in range(2):
                    u_val = u_coeffs[lr*2 + lc]
                    if u_val != 0:
                        gi = a_rows + lr
                        gj = a_cols + lc
                        U[gi*m + gj][offset + r] += u_val
                    
                    v_val = v_coeffs[lr*2 + lc]
                    if v_val != 0:
                        gj = b_rows + lr
                        gk = b_cols + lc
                        V[gj*p + gk][offset + r] += v_val
                    
                    w_val = w_coeffs[lr*2 + lc]
                    if w_val != 0:
                        gi = c_rows + lr
                        gk = c_cols + lc
                        W[gk*n + gi][offset + r] += w_val
                    
        return offset + 7

    # Part 1: 4x4 x 4x4 -> 4x4
    # A blocks:
    # A00: (0,0), A01: (0,2), A10: (2,0), A11: (2,2)
    # B blocks:
    # B00: (0,0), B01: (0,2), B10: (2,0), B11: (2,2)
    # C blocks:
    # C00: (0,0), C01: (0,2), C10: (2,0), C11: (2,2)
    
    # M1: (A00+A11) * (B00+B11) -> C00(+1), C11(+1)
    # We need to decompose (A00+A11) * (B00+B11).
    # This is a product of two linear combinations.
    # Let X = A00+A11, Y = B00+B11.
    # X is a 2x2 matrix. Y is a 2x2 matrix.
    # We apply Strassen to X*Y.
    # The inputs to Strassen are not single blocks, but sums of blocks.
    # My helper `add_2x2_strassen` assumes the inputs are single 2x2 blocks.
    # I need to modify the approach.
    
    # Instead of calling `add_2x2_strassen` with block offsets,
    # I should construct the specific u, v, w vectors for the 7 products of the 4x4 Strassen.
    
    # Let's define the 7 products of the 4x4 Strassen (M1..M7) and their Strassen sub-products.
    
    # This is getting complex. Let's just generate the U, V, W entries directly for each of the 49 terms.
    
    # Term r (0..48)
    # r // 7 -> which M (0..6)
    # r % 7 -> which Strassen sub-term (0..6)
    
    # For each M (0..6), we have linear combinations of A-blocks and B-blocks.
    # Let's define the coefficients for A-blocks and B-blocks for each M.
    
    # M0 (P1): A: [1, 0, 0, 1] (A00+A11), B: [1, 0, 0, 1] (B00+B11)
    # M1 (P2): A: [0, 0, 1, 1] (A10+A11), B: [1, 0, 0, 0] (B00)
    # M2 (P3): A: [1, 0, 0, 0] (A00), B: [0, 0, 1, -1] (B10-B11)
    # M3 (P4): A: [0, 0, 0, 1] (A11), B: [0, 1, -1, 0] (B01-B00)
    # M4 (P5): A: [1, 1, 0, 0] (A00+A01), B: [0, 0, 0, 1] (B11)
    # M5 (P6): A: [0, 0, 1, -1] (A10-A00), B: [1, 1, 0, 0] (B00+B01)
    # M6 (P7): A: [0, 1, 0, -1] (A01-A11), B: [0, 0, 1, 1] (B10+B11)
    
    # For each M, the output C is a linear combination of C-blocks.
    # C00 += 1*M0 - 1*M3 + 1*M4 - 1*M6
    # C01 += 1*M2 + 1*M3
    # C10 += 1*M1 + 1*M2
    # C11 += 1*M0 - 1*M1 + 1*M5 - 1*M2
    
    # So for M0, the W coefficients for the 7 sub-terms are:
    # W_sub_r = (coeff_in_C00 * w00) , (coeff_in_C01 * w01) , ...
    
    # Let's precompute the "effective" w coefficients for each of the 49 terms.
    
    c_coeffs = {
        0: [1, 0, 0, 1], # M0 -> C00(+1), C11(+1)
        1: [0, 0, 1, -1], # M1 -> C10(+1), C11(-1)
        2: [0, 1, 1, -1], # M2 -> C01(+1), C10(+1), C11(-1)
        3: [-1, 1, 0, 0], # M3 -> C00(-1), C01(+1)
        4: [1, 0, 0, 0], # M4 -> C00(+1)
        5: [0, 0, 0, 1], # M5 -> C11(+1)
        6: [-1, 0, 0, 0]  # M6 -> C00(-1)
    }
    
    a_block_coeffs = {
        0: [1, 0, 0, 1],
        1: [0, 0, 1, 1],
        2: [1, 0, 0, 0],
        3: [0, 0, 0, 1],
        4: [1, 1, 0, 0],
        5: [0, 0, 1, -1],
        6: [0, 1, 0, -1]
    }
    
    b_block_coeffs = {
        0: [1, 0, 0, 1],
        1: [1, 0, 0, 0],
        2: [0, 0, 1, -1],
        3: [0, 1, -1, 0],
        4: [0, 0, 0, 1],
        5: [1, 1, 0, 0],
        6: [0, 0, 1, 1]
    }
    
    # Global block indices for A, B, C (4 blocks each)
    # 0: (0,0), 1: (0,2), 2: (2,0), 3: (2,2)
    # Block size 2x2.
    
    # For each term r in 0..48:
    for r in range(49):
        m_idx = r // 7
        s_idx = r % 7
        
        # Get Strassen factors for s_idx
        u_s, v_s, w_s = strassen_factors[s_idx]
        
        # u_s is coeffs for the 2x2 input X of M_m_idx
        # v_s is coeffs for the 2x2 input Y of M_m_idx
        # w_s is coeffs for the 2x2 output Z of M_m_idx
        
        # But X is a linear combo of A-blocks.
        # Y is a linear combo of B-blocks.
        # Z is a linear combo of C-blocks.
        
        # We need to expand these.
        
        # For each local element in the 2x2 X (lr, lc):
        # Coeff is u_s[lr*2+lc]
        # This coefficient applies to the entire linear combo of A-blocks for X.
        # So for each A-block b (0..3):
        # Global A element (br*2+lr, bc*2+lc) gets coeff u_s[...] * a_block_coeffs[m_idx][b]
        
        # Similarly for B.
        
        # For C, the output of M_m_idx is Z.
        # Z is added to C-blocks with coefficients from c_coeffs[m_idx].
        # The local element of Z (lr, lc) has coeff w_s[lr*2+lc].
        # So for each C-block b (0..3):
        # Global C element (br*2+lr, bc*2+lc) gets coeff w_s[...] * c_coeffs[m_idx][b]
        
        # Let's implement this expansion.
        
        # A-blocks start positions:
        # 0: (0,0), 1: (0,2), 2: (2,0), 3: (2,2)
        # We iterate over the 4 A-blocks.
        
        for b_a in range(4):
            row_off = (b_a // 2) * 2 # 0 for top, 2 for bottom?
            # b_a: 0->(0,0), 1->(0,2), 2->(2,0), 3->(2,2)
            # Row offset: 0 if b_a < 2 else 2
            # Col offset: 0 if b_a % 2 == 0 else 2
            r_off = 0 if b_a < 2 else 2
            c_off = 0 if b_a % 2 == 0 else 2
            
            coeff_a = a_block_coeffs[m_idx][b_a]
            if coeff_a == 0: continue
            
            for lr in range(2):
                for lc in range(2):
                    val = u_s[lr*2+lc] * coeff_a
                    if val == 0: continue
                    gi = r_off + lr
                    gj = c_off + lc
                    U[gi*m + gj][r] += val
                    
        for b_b in range(4):
            r_off = 0 if b_b < 2 else 2
            c_off = 0 if b_b % 2 == 0 else 2
            coeff_b = b_block_coeffs[m_idx][b_b]
            if coeff_b == 0: continue
            
            for lr in range(2):
                for lc in range(2):
                    val = v_s[lr*2+lc] * coeff_b
                    if val == 0: continue
                    gj = r_off + lr
                    gk = c_off + lc
                    V[gj*p + gk][r] += val
                    
        for b_c in range(4):
            r_off = 0 if b_c < 2 else 2
            c_off = 0 if b_c % 2 == 0 else 2
            coeff_c = c_coeffs[m_idx][b_c]
            if coeff_c == 0: continue
            
            for lr in range(2):
                for lc in range(2):
                    val = w_s[lr*2+lc] * coeff_c
                    if val == 0: continue
                    gi = r_off + lr
                    gk = c_off + lc
                    W[gk*n + gi][r] += val

    rank_offset = 49

    # Part 2: 4x4 x 4x2 -> 4x2
    # D is 4x2 (cols 4,5 of B).
    # D_top: (0,4) to (1,5) -> 2x2
    # D_bot: (2,4) to (3,5) -> 2x2
    
    # C_right is 4x2 (cols 4,5 of C).
    # C_top_right: (0,4) to (1,5)
    # C_bot_right: (2,4) to (3,5)
    
    # Formulas:
    # C_top_right = A00 D_top + A01 D_bot
    # C_bot_right = A10 D_top + A11 D_bot
    
    # 4 products of 2x2 matrices.
    # Each product uses Strassen (rank 7).
    # Total 28 rank.
    
    products = [
        # (A_block, D_block, C_block)
        (0, 0, 0), # A00 * D_top -> C_top_right
        (1, 1, 0), # A01 * D_bot -> C_top_right
        (2, 0, 1), # A10 * D_top -> C_bot_right
        (3, 1, 1)  # A11 * D_bot -> C_bot_right
    ]
    
    for prod_idx, (a_b, d_b, c_b) in enumerate(products):
        # a_b: 0->A00, 1->A01, 2->A10, 3->A11
        # d_b: 0->D_top, 1->D_bot
        # c_b: 0->C_top_right, 1->C_bot_right
        
        # Get offsets
        a_r_off = 0 if a_b < 2 else 2
        a_c_off = 0 if a_b % 2 == 0 else 2
        
        d_r_off = 0 if d_b == 0 else 2
        d_c_off = 4 # D is at cols 4,5
        
        c_r_off = 0 if c_b == 0 else 2
        c_c_off = 4 # C_right is at cols 4,5
        
        # Add Strassen for this single 2x2 * 2x2
        # We use the same strassen_factors
        for s_idx in range(7):
            r = rank_offset + prod_idx * 7 + s_idx
            u_s, v_s, w_s = strassen_factors[s_idx]
            
            for lr in range(2):
                for lc in range(2):
                    u_val = u_s[lr*2+lc]
                    if u_val != 0:
                        gi = a_r_off + lr
                        gj = a_c_off + lc
                        U[gi*m + gj][r] += u_val
                        
                    v_val = v_s[lr*2+lc]
                    if v_val != 0:
                        gj = d_r_off + lr
                        gk = d_c_off + lc
                        V[gj*p + gk][r] += v_val
                        
                    w_val = w_s[lr*2+lc]
                    if w_val != 0:
                        gi = c_r_off + lr
                        gk = c_c_off + lc
                        W[gk*n + gi][r] += w_val
                        
    rank_offset += 28
    
    return {"U": U, "V": V, "W": W}
