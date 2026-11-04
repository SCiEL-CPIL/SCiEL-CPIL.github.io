    def add_2x2_strassen(u_offset, v_offset, w_offset, u_blocks, v_blocks, w_blocks):
        # u_blocks: list of (local_row, local_col, coeff) for U part
        # v_blocks: list of (local_row, local_col, coeff) for V part
        # w_blocks: list of (local_row, local_col, coeff) for W part
        # This adds 7 rank-1 terms to the global U, V, W.
        
        # The 7 Strassen products for 2x2:
        # M1 = (A00+A11)(B00+B11) -> C00, C11
        # M2 = (A10+A11)(B00)     -> C10, C11 (Wait, standard Strassen M2 is (A10+A11)(B01+B11)? No.)
        
        # Let's use the standard verified Strassen formulas:
        # M1 = (A00 + A11)(B00 + B11)
        # M2 = (A10 + A11)(B01 + B11)
        # M3 = (A00 - A10)(B10 + B11)
        # M4 = (A00 + A11)(B00 - B11)
        # M5 = (A00 + A10)(B00 + B10)
        # M6 = (A01 - A11)(B01 + B11)
        # M7 = (A01 - A00)(B00 + B10)
        
        # C00 = M1 - M4 + M5 - M7
        # C01 = M4 + M6
        # C10 = M3 + M5
        # C11 = M1 - M3 + M6 - M2
        
        # Each Mi is a 2x2 matrix product.
        # We need to expand each Mi into 7 rank-1 terms.
        # This is getting nested.
        
        # Alternative: Just hardcode the 49 terms for 4x4x4?
        # Or use a recursive approach.
        
        pass
    