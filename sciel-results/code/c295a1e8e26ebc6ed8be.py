def construct():
    n, m, p = 4, 4, 4
    rank = 49
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # A indices: i*4+j, i,j in 0..3
    # B indices: j*4+k, j,k in 0..3
    # C indices: k*4+i, k,i in 0..3
    
    # Define A blocks
    # A11: rows 0-1, cols 0-1
    # A12: rows 0-1, cols 2-3
    # A21: rows 2-3, cols 0-1
    # A22: rows 2-3, cols 2-3
    
    # B blocks
    # B11: rows 0-1, cols 0-1
    # B12: rows 0-1, cols 2-3
    # B21: rows 2-3, cols 0-1
    # B22: rows 2-3, cols 2-3
    
    # C blocks
    # C11: rows 0-1, cols 0-1
    # C12: rows 0-1, cols 2-3
    # C21: rows 2-3, cols 0-1
    # C22: rows 2-3, cols 2-3
    
    # We'll build the 49 rank-1 terms
    # Each term: (A_combo, B_combo, C_combo)
    # where A_combo is a dict {A_index: coeff}, etc.
    
    terms = []
    
    # Helper to add a term
    def add_term(a_combo, b_combo, c_combo):
        terms.append((a_combo, b_combo, c_combo))
    
    # Level 1 Strassen products
    # Each is a 2x2 matrix product: X * Y
    # We need to express X and Y as 2x2 matrices of A/B linear combinations
    
    # For each level-1 product, we'll compute the 7 level-2 Strassen scalar products
    
    # Let's define a function to compute Strassen for two 2x2 matrices
    # where each entry is a linear combination (dict)
    
    def lc_add(a, b):
        result = dict(a)
        for k, v in b.items():
            result[k] = result.get(k, 0) + v
        return result
    
    def lc_sub(a, b):
        result = dict(a)
        for k, v in b.items():
            result[k] = result.get(k, 0) - v
        return result
    
    def lc_scale(a, s):
        return {k: v * s for k, v in a.items()}
    
    # Define A block entries as single-element dicts
    # A11 = [[A[0,0], A[0,1]], [A[1,0], A[1,1]]]
    A11 = [[{0*4+0: 1}, {0*4+1: 1}], [{1*4+0: 1}, {1*4+1: 1}]]
    A12 = [[{0*4+2: 1}, {0*4+3: 1}], [{1*4+2: 1}, {1*4+3: 1}]]
    A21 = [[{2*4+0: 1}, {2*4+1: 1}], [{3*4+0: 1}, {3*4+1: 1}]]
    A22 = [[{2*4+2: 1}, {2*4+3: 1}], [{3*4+2: 1}, {3*4+3: 1}]]
    
    # B block entries
    B11 = [[{0*4+0: 1}, {0*4+1: 1}], [{1*4+0: 1}, {1*4+1: 1}]]
    B12 = [[{0*4+2: 1}, {0*4+3: 1}], [{1*4+2: 1}, {1*4+3: 1}]]
    B21 = [[{2*4+0: 1}, {2*4+1: 1}], [{3*4+0: 1}, {3*4+1: 1}]]
    B22 = [[{2*4+2: 1}, {2*4+3: 1}], [{3*4+2: 1}, {3*4+3: 1}]]
    
    # C block entries (for output)
    # C[i,k] has index k*4+i
    C11 = [[{0*4+0: 1}, {0*4+1: 1}], [{1*4+0: 1}, {1*4+1: 1}]]
    C12 = [[{0*4+2: 1}, {0*4+3: 1}], [{1*4+2: 1}, {1*4+3: 1}]]
    C21 = [[{2*4+0: 1}, {2*4+1: 1}], [{3*4+0: 1}, {3*4+1: 1}]]
    C22 = [[{2*4+2: 1}, {2*4+3: 1}], [{3*4+2: 1}, {3*4+3: 1}]]
    
    # Now define the 7 level-1 Strassen products
    # M1 = (A11 + A22)(B11 + B22)
    # M2 = (A21 + A22)B11
    # M3 = A11(B12 - B22)
    # M4 = A22(B21 - B11)
    # M5 = (A11 + A12)B22
    # M6 = (A21 - A11)(B11 + B12)
    # M7 = (A12 - A22)(B21 + B22)
    
    def block_add(X, Y):
        return [[lc_add(X[i][j], Y[i][j]) for j in range(2)] for i in range(2)]
    
    def block_sub(X, Y):
        return [[lc_sub(X[i][j], Y[i][j]) for j in range(2)] for i in range(2)]
    
    # Compute the 7 level-1 products' operands
    M1_X = block_add(A11, A22)
    M1_Y = block_add(B11, B22)
    
    M2_X = block_add(A21, A22)
    M2_Y = B11
    
    M3_X = A11
    M3_Y = block_sub(B12, B22)
    
    M4_X = A22
    M4_Y = block_sub(B21, B11)
    
    M5_X = block_add(A11, A12)
    M5_Y = B22
    
    M6_X = block_sub(A21, A11)
    M6_Y = block_add(B11, B12)
    
    M7_X = block_sub(A12, A22)
    M7_Y = block_add(B21, B22)
    
    # For each level-1 product, compute the 7 level-2 Strassen scalar products
    # and the resulting 2x2 matrix
    
    def strassen_2x2(X, Y):
        """Compute Strassen products for 2x2 matrices X, Y (entries are lc dicts).
        Returns list of 7 (a_combo, b_combo) pairs and the resulting 2x2 matrix
        as a list of 4 lc dicts (each being a linear combination of the 7 products).
        """
        s1 = lc_add(X[0][0], X[1][1])
        s2 = lc_add(X[1][0], X[1][1])
        s3 = X[0][0]
        s4 = X[1][1]
        s5 = lc_add(X[0][0], X[0][1])
        s6 = lc_sub(X[1][0], X[0][0])
        s7 = lc_sub(X[0][1], X[1][1])
        
        t1 = lc_add(Y[0][0], Y[1][1])
        t2 = Y[0][0]
        t3 = lc_sub(Y[0][1], Y[1][1])
        t4 = lc_sub(Y[1][0], Y[0][0])
        t5 = Y[1][1]
        t6 = lc_add(Y[0][0], Y[0][1])
        t7 = lc_add(Y[1][0], Y[1][1])
        
        products = [
            (s1, t1),  # M1
            (s2, t2),  # M2
            (s3, t3),  # M3
            (s4, t4),  # M4
            (s5, t5),  # M5
            (s6, t6),  # M6
            (s7, t7),  # M7
        ]
        
        # Result:
        # p11 = M1 + M4 - M5 + M7
        # p12 = M3 + M5
        # p21 = M2 + M4
        # p22 = M1 - M2 + M3 + M6
        
        # We'll return the products and a mapping from result position to coefficients
        # For each result position, we need to know which products contribute with what coefficient
        result_coeffs = [
            [(0, 1), (3, 1), (4, -1), (6, 1)],  # p11
            [(2, 1), (4, 1)],                    # p12
            [(1, 1), (3, 1)],                    # p21
            [(0, 1), (1, -1), (2, 1), (5, 1)],  # p22
        ]
        
        return products, result_coeffs
    
    # Now build all 49 terms
    # For each level-1 product, compute its 7 level-2 products
    # Then for each level-2 product, add a rank-1 term
    # The output contribution comes from the level-1 result combination
    
    level1_products = [
        (M1_X, M1_Y, "M1"),
        (M2_X, M2_Y, "M2"),
        (M3_X, M3_Y, "M3"),
        (M4_X, M4_Y, "M4"),
        (M5_X, M5_Y, "M5"),
        (M6_X, M6_Y, "M6"),
        (M7_X, M7_Y, "M7"),
    ]
    
    # Level-1 result combinations:
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    # For each level-1 product Mi, for each of its 7 level-2 products,
    # we add a term. The output coefficient for that term is determined by
    # which C-blocks Mi contributes to, and within those blocks, which position.
    
    # Let me restructure: for each level-1 product, I'll compute its 7 scalar products,
    # and for each scalar product, I'll determine the output contribution.
    
    # Actually, it's cleaner to: for each level-1 product Mi, compute its 7 level-2 products.
    # Each level-2 product is a scalar s*t. This scalar contributes to Mi (the 2x2 matrix).
    # Mi then contributes to certain C-blocks with certain coefficients.
    # So the total output contribution of a level-2 scalar product is the sum over
    # all C-block positions where Mi appears, of (coefficient in Mi's result) * (coefficient in C-block combination).
    
    # Let me just iterate over all 49 terms and build U, V, W.
    
    term_idx = 0
    for (X, Y, name) in level1_products:
        products, result_coeffs = strassen_2x2(X, Y)
        
        # result_coeffs[0] = p11 coeffs, [1] = p12, [2] = p21, [3] = p22
        # Each is a list of (product_idx, coeff)
        
        # For each of the 7 products, determine its output contribution
        # product_idx in 0..6 corresponds to the 7 Strassen products
        
        # The 2x2 result matrix has positions:
        # p11 -> C-block position (0,0)
        # p12 -> C-block position (0,1)
        # p21 -> C-block position (1,0)
        # p22 -> C-block position (1,1)
        
        # For this level-1 product, determine which C-blocks it contributes to
        # and with what coefficient
        
        # This depends on which level-1 product it is (M1..M7)
        # Let me define the level-1 combination:
        # C11 = M1 + M4 - M5 + M7
        # C12 = M3 + M5
        # C21 = M2 + M4
        # C22 = M1 - M2 + M3 + M6
        
        # So for each Mi, I need to know which C-blocks it appears in and with what coefficient.
        
        # Let me define this as a dict: name -> list of (C_block_name, coeff)
        # C_block_name: "C11", "C12", "C21", "C22"
        
        level1_combos = {
            "M1": [("C11", 1), ("C22", 1)],
            "M2": [("C21", 1), ("C22", -1)],
            "M3": [("C12", 1), ("C22", 1)],
            "M4": [("C11", 1), ("C21", 1)],
            "M5": [("C11", -1), ("C12", 1)],
            "M6": [("C22", 1)],
            "M7": [("C11", 1)],
        }
        
        # For each level-2 product (scalar s*t), the output is:
        # sum over all (C_block, coeff1) in level1_combos[name] of
        #   coeff1 * (sum over (prod_idx, coeff2) in result_coeffs[position] where prod_idx == this_product_idx)
        #   * C_block[position]
        
        # Wait, I need to be more careful. The level-2 product at index pi contributes to
        # the level-1 result matrix at certain positions with certain coefficients.
        # Then the level-1 result matrix contributes to C-blocks with certain coefficients.
        
        # Let me just compute for each level-2 product pi, the total output lc dict.
        
        for pi in range(7):
            a_combo, b_combo = products[pi]
            
            # Determine which positions in the level-1 result this product contributes to
            # result_coeffs[pos] is a list of (prod_idx, coeff)
            # So for product pi, find all positions where pi appears
            
            output_lc = {}
            for pos in range(4):
                for (prod_idx, coeff2) in result_coeffs[pos]:
                    if prod_idx == pi:
                        # This product contributes to position pos with coefficient coeff2
                        # Position pos corresponds to a C-block
                        # pos 0 -> C11, pos 1 -> C12, pos 2 -> C21, pos 3 -> C22
                        pos_to_block = ["C11", "C12", "C21", "C22"]
                        block_name = pos_to_block[pos]
                        
                        # Now find the level-1 coefficient for this block
                        for (block_name2, coeff1) in level1_combos[name]:
                            if block_name2 == block_name:
                                # Total coefficient for this C-block position
                                total_coeff = coeff1 * coeff2
                                # Add to output_lc
                                block_data = {"C11": C11, "C12": C12, "C21": C21, "C22": C22}[block_name]
                                # Position in block: pos 0 -> (0,0), pos 1 -> (0,1), pos 2 -> (1,0), pos 3 -> (1,1)
                                # But wait, pos is the position in the 2x2 result matrix
                                # For a given block, the position within the block is determined by pos
                                # pos 0: row 0, col 0
                                # pos 1: row 0, col 1
                                # pos 2: row 1, col 0
                                # pos 3: row 1, col 1
                                r = pos // 2
                                c = pos % 2
                                entry = block_data[r][c]
                                for idx, val in entry.items():
                                    output_lc[idx] = output_lc.get(idx, 0) + total_coeff * val
                        break
                        # break the inner loop since we found the match
                    # End if prod_idx == pi
                # End for (prod_idx, coeff2)
            # End for pos
            
            add_term(a_combo, b_combo, output_lc)
            term_idx += 1
    
    # Now populate U, V, W
    for r, (a_combo, b_combo, c_combo) in enumerate(terms):
        for idx, coeff in a_combo.items():
            U[idx][r] += coeff
        for idx, coeff in b_combo.items():
            V[idx][r] += coeff
        for idx, coeff in c_combo.items():
            W[idx][r] += coeff
    
    return {"U": U, "V": V, "W": W}
