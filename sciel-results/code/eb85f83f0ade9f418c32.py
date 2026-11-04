def _strassen_4x4():
    """
    Generate 49 terms for 4x4x4 matrix multiplication using recursive Strassen.
    """
    terms = []
    strassen_2x2 = _strassen_2x2()
    
    # Map local 2x2 block indices to 4x4 global indices
    # Block 0: rows 0-1, cols 0-1
    # Block 1: rows 0-1, cols 2-3
    # Block 2: rows 2-3, cols 0-1
    # Block 3: rows 2-3, cols 2-3
    
    def map_index_4x4(block_idx, local_row, local_col):
        block_row = block_idx // 2
        block_col = block_idx % 2
        global_row = block_row * 2 + local_row
        global_col = block_col * 2 + local_col
        return global_row, global_col

    for s_term in strassen_2x2:
        u_local_4, u_coeffs_4, v_local_4, v_coeffs_4, w_local_4, w_coeffs_4 = s_term
        
        # u_local_4 contains indices 0..3 representing 2x2 blocks
        # We need to expand each block index into 4 local indices (0..3)
        
        # For each sub-term in the 2x2 Strassen decomposition of the blocks
        # Note: s_term defines the linear combination of the 4 blocks.
        # We need to apply the 2x2 Strassen to each block pair?
        # No, s_term is a linear combination of BLOCKS.
        # Example: M1 = (A00+A11)(B00+B11).
        # This means we take the sum of blocks A00 and A11, multiply by sum of B00 and B11.
        # This is a matrix multiplication of two 4x4 matrices formed by these blocks?
        # No, A00 and A11 are 2x2 matrices. Their sum is a 2x2 matrix.
        # So M1 is the product of two 2x2 matrices.
        # We can compute this product using the 7-term Strassen for 2x2 matrices.
        
        # So for each s_term, we construct the "effective" 2x2 matrices for U, V, W?
        # No, the terms are sparse.
        
        # Let's iterate over the 7 sub-terms of the 2x2 Strassen multiplication
        # that computes the product of the two 2x2 matrices defined by s_term.
        
        # Actually, it's simpler to just generate all 49 combinations directly.
        # The 49 terms correspond to all pairs of (sub_term_u, sub_term_v) ?
        # No.
        
        # Let's stick to the definition:
        # M1 = (A00+A11)(B00+B11).
        # Let X = A00+A11, Y = B00+B11.
        # M1 = X*Y.
        # X is a 2x2 matrix. Y is a 2x2 matrix.
        # We decompose X*Y using _strassen_2x2().
        # Each term in _strassen_2x2() for X*Y involves linear combinations of entries of X and Y.
        # Entries of X are sums of entries of A00 and A11.
        # So each term in M1 decomposition is a linear combination of entries of A00, A11, B00, B11.
        
        # We can generate the terms by:
        # 1. For each s_term in strassen_2x2 (the outer level):
        #    - Identify which blocks are involved in U and V.
        #    - For each sub_term in strassen_2x2 (the inner level, for the 2x2 block multiplication):
        #       - Combine the coefficients.
        
        # This is getting complicated to generalize dynamically.
        # Given the time, I will implement the 49 terms explicitly or use a simpler recursive call.
        
        # Simpler approach:
        # Just use the existing schoolbook for 4x4x4 but that's rank 64.
        # Strassen 49 is better.
        
        # Let's try to construct the 49 terms by iterating over the 7 outer terms
        # and for each, iterating over the 7 inner terms.
        
        # For a given outer term (e.g. M1), we have:
        # u_blocks: list of block indices and coeffs
        # v_blocks: list of block indices and coeffs
        
        # We form U_mat (2x2) and V_mat (2x2) where U_mat[r,c] = sum of U_block[r,c] * coeff.
        # Then we apply _strassen_2x2 to U_mat * V_mat.
        # The resulting terms will have U coefficients that are sums of U_block entries.
        
        # We can simulate this.
        
        u_block_indices, u_block_coeffs = zip(*[(idx, coeff) for idx, coeff in zip(u_local_4, u_coeffs_4) if coeff != 0])
        v_block_indices, v_block_coeffs = zip(*[(idx, coeff) for idx, coeff in zip(v_local_4, v_coeffs_4) if coeff != 0])
        w_block_indices, w_block_coeffs = zip(*[(idx, coeff) for idx, coeff in zip(w_local_4, w_coeffs_4) if coeff != 0])
        
        # Build the effective 2x2 matrices for the sub-multiplication
        # U_eff[li, lj] = sum of coeffs from u_block_indices
        U_eff = [[0]*2 for _ in range(2)]
        for b_idx, coeff in zip(u_block_indices, u_block_coeffs):
            br = b_idx // 2
            bc = b_idx % 2
            for lr in range(2):
                for lc in range(2):
                    U_eff[lr][lc] += coeff
        
        V_eff = [[0]*2 for _ in range(2)]
        for b_idx, coeff in zip(v_block_indices, v_block_coeffs):
            br = b_idx // 2
            bc = b_idx % 2
            for lr in range(2):
                for lc in range(2):
                    V_eff[lr][lc] += coeff
        
        # Now apply _strassen_2x2 to U_eff and V_eff
        # The output W_eff will be added to the global W
        
        for sub_term in strassen_2x2:
            sub_u_idx, sub_u_coeff, sub_v_idx, sub_v_coeff, sub_w_idx, sub_w_coeff = sub_term
            
            u_dict = {}
            v_dict = {}
            w_dict = {}
            
            # Expand sub_u_idx (0..3) to 4x4 global indices
            for li, coeff in zip(sub_u_idx, sub_u_coeff):
                if coeff == 0: continue
                lr = li // 2
                lc = li % 2
                # U_eff[lr][lc] is the coefficient for this entry in the sub-multiplication
                # But U_eff[lr][lc] is a sum of block coeffs.
                # We need to distribute this back to the original blocks.
                # This is tricky.
                
                # Alternative: Just expand the blocks directly.
                # For each block in u_block_indices:
                #   For each entry in that block:
                #     Add to u_dict with coeff = block_coeff * sub_u_coeff
                pass
            
            # This manual expansion is error-prone.
            # I will use the schoolbook for 4x4x4 (rank 64) for now to ensure correctness,
            # and then optimize if time permits.
            # Wait, rank 64 gives score 1.0.
            # Rank 49 gives score 1.136.
            # Rank 27 (if possible) gives 1.42.
            
            # Let's try to implement the 49 terms correctly.
            pass
    return terms
