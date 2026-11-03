"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Use rank 15 decomposition for 2x2 by 2x6 blocks.
    # A is 2x4, split into A1 (cols 0-1) and A2 (cols 2-3).
    # B is 4x6.
    # C = A1 * B_top + A2 * B_bottom.
    # B_top is 2x6, B_bottom is 2x6.
    # Rank of 2x2 by 2x6 is 15.
    # Total rank = 15 + 15 = 30.
    
    rank = 30
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen 7 products for 2x2 by 2x2
    strassen_defs = [
        ({0: 1, 3: 1}, {0: 1, 3: 1}), # M1
        ({2: 1, 3: 1}, {0: 1}),       # M2
        ({0: 1}, {1: 1, 3: -1}),      # M3
        ({3: 1}, {2: 1, 0: -1}),      # M4
        ({0: 1, 1: 1}, {3: 1}),       # M5
        ({2: 1, 0: -1}, {0: 1, 1: 1}),# M6
        ({1: 1, 3: -1}, {2: 1, 3: 1}) # M7
    ]
    
    res_coeffs_2x2 = [
        [1, 0, 0, 1, -1, 0, 1], # Z00
        [0, 0, 1, 0, 1, 0, 0],  # Z01
        [0, 1, 0, 1, 0, 0, 0],  # Z10
        [1, -1, 1, 0, 0, 1, 0]  # Z11
    ]
    
    # Extension for adding a column (2x1) to 2x2 by 2x2 -> 2x2 by 2x3
    # Adds 2 products.
    # For column y (indices 0,1 in local 2x1), add:
    # P1: X00 * y0
    # P2: X01 * y1
    # P3: X10 * y0
    # P4: X11 * y1
    # Wait, rank increases by 2, so we need 2 products.
    # This implies we can't just use trivial products.
    # However, for the sake of a working improvement, I will use a slightly higher rank
    # if I can't derive the 15-rank form, but 30 is the target.
    # Let's try to construct the 15-rank form.
    # Actually, let's use a known 28-rank decomposition if 15 is hard?
    # No, 30 is better than 42.
    
    # I will implement a 24-rank decomposition for 2x2 by 2x6 if possible?
    # Rank 2p+3 = 15.
    
    # Let's use the 15-rank decomposition.
    # Factors 0-6: Strassen for cols 0,1.
    # Factors 7-8: Ext for col 2.
    # Factors 9-10: Ext for col 3.
    # Factors 11-12: Ext for col 4.
    # Factors 13-14: Ext for col 5.
    
    # Extension for column c (local index 2, 3, 4, 5 in 2x6 Y)
    # Let y_c be the column.
    # We add 2 products.
    # It is known that we can use:
    # P_ext1 = (X00 + X01) * y_c0 ?
    # Let's assume the standard extension.
    
    def add_block_2x2x2x6(a_col_offset, b_row_offset, c_col_offset, start_col):
        # a_col_offset: 0 or 2 in A
        # b_row_offset: 0 or 2 in B
        # c_col_offset: 0, 2, 4 in C (but C is 2x6, so we cover all 6 cols)
        
        # 1. Strassen for cols 0,1 of the 2x6 block
        for s in range(7):
            col = start_col + s
            x_coeffs, y_coeffs = strassen_defs[s]
            
            for x_idx, val in x_coeffs.items():
                r = x_idx // 2
                c = x_idx % 2
                global_i = r
                global_j = a_col_offset + c
                U[global_i * m + global_j][col] = val
                
            for y_idx, val in y_coeffs.items():
                r = y_idx // 2
                c = y_idx % 2
                # y_idx 0,1,2,3 correspond to local 2x2 Y (cols 0,1)
                global_j = b_row_offset + r
                global_k = c_col_offset + c # Wait, c_col_offset is 0 for the first 2 cols of the block
                # The block covers 6 cols. Strassen covers first 2.
                V[global_j * p + global_k][col] = val
                
            for z_idx in range(4):
                coeff = res_coeffs_2x2[z_idx][s]
                if coeff != 0:
                    z_r = z_idx // 2
                    z_c = z_idx % 2
                    global_i = z_r
                    global_k = c_col_offset + z_c
                    W[global_k * n + global_i][col] = coeff

        # 2. Extensions for cols 2,3,4,5
        # For each new column, add 2 products.
        # Total 4 * 2 = 8 products.
        # Indices 7..14.
        
        # We need the specific linear forms for the extension.
        # Based on literature, the extension uses:
        # For column k (0-indexed in the extension part, so global col 2+k):
        # P1 = X00 * y_k0
        # P2 = X01 * y_k1
        # This is 2 products? No, that's 2 products but only covers Z0.
        # We need Z1 too.
        
        # Actually, let's just use 4 products per extension column?
        # That would be 7 + 4*4 = 23 per block. Total 46. Worse.
        
        # Let's stick to 42 if I can't find 15.
        # But I can try 28?
        
        # Let's use a simpler improvement:
        # Use Strassen for 2x2 by 2x2 for the first 2 cols (rank 7).
        # Use Trivial for the next 4 cols (rank 4 per col = 16).
        # Total 23 per block. Total 46. No.
        
        # Let's use Strassen for 2x2 by 2x4?
        # Rank 2*4+3 = 11.
        # Then 2x2 by 2x2 for last 2 cols?
        # No, it's one 2x2 by 2x6.
        
        # I will implement the 15-rank decomposition by guessing the extension forms
        # or falling back to 42 if it fails.
        # For now, I will return the 42 rank solution to be safe, 
        # but I will structure it to allow 30 later.
        
        # Actually, I can use the fact that 2x2 by 2x6 rank is 15.
        # I will hardcode the 15 forms if I can find them online.
        # Since I can't, I will use a 24-rank decomposition for 2x2 by 2x6?
        # No.
        
        # Let's try to construct the 15-rank decomposition.
        # The 2-product extension for column y:
        # P1 = (X00 + X11) y0
        # P2 = (X01 + X10) y1
        # This doesn't seem right.
        
        # I will keep the 42 rank solution for now to ensure validity.
        pass

    # Fallback to 42 rank solution
    rank = 42
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    strassen_defs = [
        ({0: 1, 3: 1}, {0: 1, 3: 1}),
        ({2: 1, 3: 1}, {0: 1}),
        ({0: 1}, {1: 1, 3: -1}),
        ({3: 1}, {2: 1, 0: -1}),
        ({0: 1, 1: 1}, {3: 1}),
        ({2: 1, 0: -1}, {0: 1, 1: 1}),
        ({1: 1, 3: -1}, {2: 1, 3: 1})
    ]
    
    res_coeffs = [
        [1, 0, 0, 1, -1, 0, 1],
        [0, 0, 1, 0, 1, 0, 0],
        [0, 1, 0, 1, 0, 0, 0],
        [1, -1, 1, 0, 0, 1, 0]
    ]
    
    def add_strassen_block(a_col_offset, b_row_offset, c_col_offset, start_col):
        for s in range(7):
            col = start_col + s
            x_coeffs, y_coeffs = strassen_defs[s]
            
            for x_idx, val in x_coeffs.items():
                r = x_idx // 2
                c = x_idx % 2
                global_i = r
                global_j = a_col_offset + c
                U[global_i * m + global_j][col] = val
                
            for y_idx, val in y_coeffs.items():
                r = y_idx // 2
                c = y_idx % 2
                global_j = b_row_offset + r
                global_k = c_col_offset + c
                V[global_j * p + global_k][col] = val
                
            for z_idx in range(4):
                coeff = res_coeffs[z_idx][s]
                if coeff != 0:
                    z_r = z_idx // 2
                    z_c = z_idx % 2
                    global_i = z_r
                    global_k = c_col_offset + z_c
                    W[global_k * n + global_i][col] = coeff

    add_strassen_block(0, 0, 0, 0)
    add_strassen_block(2, 2, 0, 7)
    add_strassen_block(0, 0, 2, 14)
    add_strassen_block(2, 2, 2, 21)
    add_strassen_block(0, 0, 4, 28)
    add_strassen_block(2, 2, 4, 35)
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
