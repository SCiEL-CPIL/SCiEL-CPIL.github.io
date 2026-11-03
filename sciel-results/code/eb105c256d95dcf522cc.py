import numpy as np

DIMENSIONS = [4, 4, 7]

def construct():
    n, m, p = DIMENSIONS
    # U: (n*m, rank)
    # V: (m*p, rank)
    # W: (p*n, rank)
    
    # We will build lists of columns for U, V, W
    U_cols = []
    V_cols = []
    W_cols = []
    
    def add_term(u_vec, v_vec, w_vec):
        U_cols.append(u_vec)
        V_cols.append(v_vec)
        W_cols.append(w_vec)
        
    # --- Part 1: 4x4x4 (Rank 49) ---
    # Strassen for 2x2x2 helper
    # Returns lists of (u, v, w) for the 7 terms
    # u, v, w are vectors of length 4
    def strassen_2x2x2():
        terms = []
        # M1
        u1 = [1, 0, 0, 1] # A00+A11
        v1 = [1, 0, 0, 1] # B00+B11
        w1 = [1, 0, 0, 1] # C00, C11
        terms.append((u1, v1, w1))
        
        # M2
        u2 = [0, 0, 1, 1] # A10+A11
        v2 = [1, 0, 0, 0] # B00
        w2 = [0, 0, 1, -1] # C10, C11
        terms.append((u2, v2, w2))
        
        # M3
        u3 = [1, 0, 0, 0] # A00
        v3 = [0, 1, 0, -1] # B01-B11
        w3 = [0, 1, 0, 1] # C01, C11
        terms.append((u3, v3, w3))
        
        # M4
        u4 = [0, 0, 0, 1] # A11
        v4 = [0, 0, 1, -1] # B10-B00
        w4 = [1, 0, 1, 0] # C00, C10
        terms.append((u4, v4, w4))
        
        # M5
        u5 = [1, 1, 0, 0] # A00+A01
        v5 = [0, 0, 0, 1] # B11
        w5 = [-1, 1, 0, 0] # C00, C01
        terms.append((u5, v5, w5))
        
        # M6
        u6 = [-1, 0, 1, 0] # A10-A00
        v6 = [1, 1, 0, 0] # B00+B01
        w6 = [0, 0, 0, -1] # C11
        terms.append((u6, v6, w6))
        
        # M7
        u7 = [0, 1, 0, -1] # A01-A11
        v7 = [0, 0, 1, 1] # B10+B11
        w7 = [1, 0, 0, 0] # C00
        terms.append((u7, v7, w7))
        
        return terms

    strassen_terms = strassen_2x2x2()
    
    # Map block indices to global indices
    # A blocks:
    # A1: rows 0,1 cols 0,1 -> global 0,1,4,5
    # A2: rows 0,1 cols 2,3 -> global 2,3,6,7
    # A3: rows 2,3 cols 0,1 -> global 8,9,12,13
    # A4: rows 2,3 cols 2,3 -> global 10,11,14,15
    
    # B blocks:
    # B1: rows 0,1 cols 0,1 -> global 0,1,4,5
    # B2: rows 0,1 cols 2,3 -> global 2,3,6,7
    # B3: rows 2,3 cols 0,1 -> global 8,9,12,13
    # B4: rows 2,3 cols 2,3 -> global 10,11,14,15
    
    # C blocks:
    # C1: rows 0,1 cols 0,1 -> global 0,1,4,5
    # C2: rows 0,1 cols 2,3 -> global 2,3,6,7
    # C3: rows 2,3 cols 0,1 -> global 8,9,12,13
    # C4: rows 2,3 cols 2,3 -> global 10,11,14,15
    
    # Strassen products:
    # M1 = (A1+A4)(B1+B4)
    # M2 = (A3+A4)B1
    # M3 = A1(B2-B4)
    # M4 = A4(B3-B1)
    # M5 = (A1+A2)B4
    # M6 = (A3-A1)(B1+B2)
    # M7 = (A2-A4)(B3+B4)
    
    # We need to form the U, V vectors for each product, then decompose the resulting 2x2x2 product.
    
    def expand_2x2_block(block_idx, local_vec, global_size):
        # block_idx: 0,1,2,3 corresponding to A1,A2,A3,A4
        # local_vec: length 4
        # returns global vector of length 16
        g = [0] * global_size
        # Mapping for A:
        # 0: 0,1,4,5
        # 1: 2,3,6,7
        # 2: 8,9,12,13
        # 3: 10,11,14,15
        maps = [
            [0, 1, 4, 5],
            [2, 3, 6, 7],
            [8, 9, 12, 13],
            [10, 11, 14, 15]
        ]
        for i, val in enumerate(local_vec):
            if val != 0:
                g[maps[block_idx][i]] += val
        return g

    # For each Strassen product M_i, we have a linear combination of A blocks and B blocks.
    # We form the "effective" A and B vectors for the 2x2x2 Strassen decomposition.
    
    # M1: A = A1+A4, B = B1+B4
    u_m1 = [0]*16
    v_m1 = [0]*16
    u_m1 = [x+y for x,y in zip(expand_2x2_block(0, [1,0,0,0], 16), expand_2x2_block(3, [1,0,0,0], 16))]
    v_m1 = [x+y for x,y in zip(expand_2x2_block(0, [1,0,0,0], 16), expand_2x2_block(3, [1,0,0,0], 16))]
    # Wait, the Strassen decomposition acts on the 2x2 blocks.
    # The "A" in the 2x2x2 Strassen is the 2x2 block itself.
    # So if the product is (A1+A4)(B1+B4), the "A" matrix for the inner Strassen is A1+A4.
    # We need to feed the coefficients of A1, A4 into the U vector of the inner Strassen.
    
    # Let's refine the approach.
    # For each of the 7 outer Strassen products, we identify which A blocks and B blocks are involved.
    # Then we sum the local Strassen U, V, W vectors scaled by the coefficients.
    
    # Outer Strassen Products:
    # M1: A = A1+A4, B = B1+B4
    # M2: A = A3+A4, B = B1
    # M3: A = A1, B = B2-B4
    # M4: A = A4, B = B3-B1
    # M5: A = A1+A2, B = B4
    # M6: A = A3-A1, B = B1+B2
    # M7: A = A2-A4, B = B3+B4
    
    # For each product, we iterate through the 7 inner Strassen terms.
    
    for prod_idx in range(7):
        # Determine A and B block combinations
        # a_coeffs: list of (block_idx, coeff)
        # b_coeffs: list of (block_idx, coeff)
        
        if prod_idx == 0: # M1
            a_coeffs = [(0, 1), (3, 1)]
            b_coeffs = [(0, 1), (3, 1)]
        elif prod_idx == 1: # M2
            a_coeffs = [(2, 1), (3, 1)]
            b_coeffs = [(0, 1)]
        elif prod_idx == 2: # M3
            a_coeffs = [(0, 1)]
            b_coeffs = [(1, 1), (3, -1)]
        elif prod_idx == 3: # M4
            a_coeffs = [(3, 1)]
            b_coeffs = [(2, 1), (0, -1)]
        elif prod_idx == 4: # M5
            a_coeffs = [(0, 1), (1, 1)]
            b_coeffs = [(3, 1)]
        elif prod_idx == 5: # M6
            a_coeffs = [(2, 1), (0, -1)]
            b_coeffs = [(0, 1), (1, 1)]
        elif prod_idx == 6: # M7
            a_coeffs = [(1, 1), (3, -1)]
            b_coeffs = [(2, 1), (3, 1)]
            
        # For each inner Strassen term
        for inner_idx in range(7):
            u_inner, v_inner, w_inner = strassen_terms[inner_idx]
            
            u_global = [0] * 16
            v_global = [0] * 16
            w_global = [0] * 16
            
            # Expand U
            for b_idx, coeff in a_coeffs:
                # u_inner is local to the 2x2 block
                # We need to add coeff * expand(u_inner, b_idx)
                exp = expand_2x2_block(b_idx, u_inner, 16)
                for i in range(16):
                    u_global[i] += coeff * exp[i]
                    
            # Expand V
            for b_idx, coeff in b_coeffs:
                exp = expand_2x2_block(b_idx, v_inner, 16)
                for i in range(16):
                    v_global[i] += coeff * exp[i]
                    
            # Expand W
            # The output C block is the same index as the A/B blocks?
            # No, the output C is also a 2x2 block.
            # The Strassen product M_i contributes to specific C blocks.
            # In the outer Strassen, M1 contributes to C1, C2, C3, C4.
            # Specifically:
            # C1 = M1 + M4 - M5 + M7
            # C2 = M3 + M5
            # C3 = M2 + M4
            # C4 = M1 - M2 + M3 - M6
            
            # So for a given outer product M_i, we need to know which C blocks it contributes to and with what sign.
            
            c_contribs = []
            if prod_idx == 0: # M1
                c_contribs = [(0, 1), (3, 1)] # C1, C4
            elif prod_idx == 1: # M2
                c_contribs = [(2, 1), (3, -1)] # C3, C4
            elif prod_idx == 2: # M3
                c_contribs = [(1, 1), (3, 1)] # C2, C4
            elif prod_idx == 3: # M4
                c_contribs = [(0, 1), (2, 1)] # C1, C3
            elif prod_idx == 4: # M5
                c_contribs = [(0, -1), (1, 1)] # C1, C2
            elif prod_idx == 5: # M6
                c_contribs = [(3, -1)] # C4
            elif prod_idx == 6: # M7
                c_contribs = [(0, 1)] # C1
                
            for c_idx, coeff in c_contribs:
                exp = expand_2x2_block(c_idx, w_inner, 16)
                for i in range(16):
                    w_global[i] += coeff * exp[i]
                    
            add_term(u_global, v_global, w_global)

    # --- Part 2: 4x4x3 (Rank 44) ---
    # For each of the 4 A blocks (A1..A4):
    #   B block is 2x3.
    #   Split B into B_left (2x2) and B_right (2x1).
    #   C_left = A_block * B_left (Rank 7)
    #   C_right = A_block * B_right (Rank 4)
    
    def schoolbook_2x2x1():
        terms = []
        # C00 = A00B00 + A01B10
        # C10 = A10B00 + A11B10
        # U: A, V: B (2x1), W: C (2x1)
        # Indices: U 0..3, V 0..1, W 0..1
        
        # Term 1: A00 B00 -> C00
        u1 = [1, 0, 0, 0]
        v1 = [1, 0]
        w1 = [1, 0]
        terms.append((u1, v1, w1))
        
        # Term 2: A01 B10 -> C00
        u2 = [0, 1, 0, 0]
        v2 = [0, 1]
        w2 = [1, 0]
        terms.append((u2, v2, w2))
        
        # Term 3: A10 B00 -> C10
        u3 = [0, 0, 1, 0]
        v3 = [1, 0]
        w3 = [0, 1]
        terms.append((u3, v3, w3))
        
        # Term 4: A11 B10 -> C10
        u4 = [0, 0, 0, 1]
        v4 = [0, 1]
        w4 = [0, 1]
        terms.append((u4, v4, w4))
        
        return terms

    schoolbook_terms = schoolbook_2x2x1()
    
    # Global V indices for Part 2:
    # B is 4x7. Columns 4,5,6.
    # Block B1 (rows 0,1): cols 4,5,6 -> global V indices:
    #   B1_left (cols 4,5): 0*7+4, 0*7+5, 1*7+4, 1*7+5 -> 4,5,11,12
    #   B1_right (col 6): 0*7+6, 1*7+6 -> 6,13
    # Block B2 (rows 0,1): Wait, A is 4x4. B is 4x7.
    # A rows 0,1 multiply B rows 0,1.
    # A rows 2,3 multiply B rows 2,3.
    
    # So:
    # A1 (rows 0,1 cols 0,1) multiplies B1 (rows 0,1 cols 4,5,6)
    # A2 (rows 0,1 cols 2,3) multiplies B1 (rows 0,1 cols 4,5,6)
    # A3 (rows 2,3 cols 0,1) multiplies B2 (rows 2,3 cols 4,5,6)
    # A4 (rows 2,3 cols 2,3) multiplies B2 (rows 2,3 cols 4,5,6)
    
    # Wait, the multiplication is C = A B.
    # C[i,k] = sum_j A[i,j] B[j,k].
    # A is 4x4. B is 4x7.
    # A1 (top-left 2x2) multiplies B_top (top 2 rows of B).
    # A2 (top-right 2x2) multiplies B_top (top 2 rows of B).
    # A3 (bottom-left 2x2) multiplies B_bottom (bottom 2 rows of B).
    # A4 (bottom-right 2x2) multiplies B_bottom (bottom 2 rows of B).
    
    # So B1 is B_top (rows 0,1, cols 4,5,6).
    # B2 is B_bottom (rows 2,3, cols 4,5,6).
    
    # We need to map the local V indices (0,1 for left, 0,1 for right) to global V indices.
    
    def expand_v_block(b_row_offset, local_vec, global_size):
        # b_row_offset: 0 for top, 1 for bottom
        # local_vec: length 3 (for 2x3) or split
        # We handle left and right separately.
        pass

    # Let's just manually map the 4 blocks.
    
    # Block 1: A1 * B_top
    # A1 global U: 0,1,4,5
    # B_top global V:
    #   Left (cols 4,5): 4,5,11,12
    #   Right (col 6): 6,13
    
    # Block 2: A2 * B_top
    # A2 global U: 2,3,6,7
    # B_top global V: same as above
    
    # Block 3: A3 * B_bottom
    # A3 global U: 8,9,12,13
    # B_bottom global V:
    #   Left (cols 4,5): 18,19,25,26
    #   Right (col 6): 20,27
    
    # Block 4: A4 * B_bottom
    # A4 global U: 10,11,14,15
    # B_bottom global V: same as above
    
    # C blocks:
    # C1 (rows 0,1 cols 4,5,6)
    # C2 (rows 0,1 cols 4,5,6) -> Wait, C is 4x7.
    # C1 is top-left 2x3? No.
    # C = A B.
    # C rows 0,1 come from A rows 0,1.
    # C rows 2,3 come from A rows 2,3.
    
    # So:
    # A1, A2 contribute to C rows 0,1.
    # A3, A4 contribute to C rows 2,3.
    
    # C_top (rows 0,1, cols 4,5,6)
    # C_bottom (rows 2,3, cols 4,5,6)
    
    # Global W indices:
    # C_top:
    #   Left (cols 4,5): 4*4+0, 4*4+1, 5*4+0, 5*4+1 -> 16,17,20,21
    #   Right (col 6): 6*4+0, 6*4+1 -> 24,25
    # C_bottom:
    #   Left (cols 4,5): 4*4+2, 4*4+3, 5*4+2, 5*4+3 -> 18,19,22,23
    #   Right (col 6): 6*4+2, 6*4+3 -> 26,27
    
    # We will process each of the 4 A blocks.
    
    for a_block_idx in range(4):
        # Determine U global map
        u_maps = [
            [0, 1, 4, 5],
            [2, 3, 6, 7],
            [8, 9, 12, 13],
            [10, 11, 14, 15]
        ]
        u_map = u_maps[a_block_idx]
        
        # Determine B and C global maps
        if a_block_idx < 2:
            # Top
            v_left_map = [4, 5, 11, 12]
            v_right_map = [6, 13]
            w_left_map = [16, 17, 20, 21]
            w_right_map = [24, 25]
        else:
            # Bottom
            v_left_map = [18, 19, 25, 26]
            v_right_map = [20, 27]
            w_left_map = [18, 19, 22, 23]
            w_right_map = [26, 27]
            
        # Process Strassen for Left (2x2x2)
        for inner_idx in range(7):
            u_inner, v_inner, w_inner = strassen_terms[inner_idx]
            
            u_global = [0] * 16
            v_global = [0] * 28
            w_global = [0] * 28
            
            for i in range(4):
                if u_inner[i] != 0:
                    u_global[u_map[i]] = u_inner[i]
                if v_inner[i] != 0:
                    v_global[v_left_map[i]] = v_inner[i]
                if w_inner[i] != 0:
                    w_global[w_left_map[i]] = w_inner[i]
                    
            add_term(u_global, v_global, w_global)
            
        # Process Schoolbook for Right (2x2x1)
        for inner_idx in range(4):
            u_inner, v_inner, w_inner = schoolbook_terms[inner_idx]
            
            u_global = [0] * 16
            v_global = [0] * 28
            w_global = [0] * 28
            
            for i in range(4):
                if u_inner[i] != 0:
                    u_global[u_map[i]] = u_inner[i]
                    
            # v_inner is length 2
            for i in range(2):
                if v_inner[i] != 0:
                    v_global[v_right_map[i]] = v_inner[i]
                    
            # w_inner is length 2
            for i in range(2):
                if w_inner[i] != 0:
                    w_global[w_right_map[i]] = w_inner[i]
                    
            add_term(u_global, v_global, w_global)

    # Convert lists to matrices
    U = np.array(U_cols).T
    V = np.array(V_cols).T
    W = np.array(W_cols).T
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
