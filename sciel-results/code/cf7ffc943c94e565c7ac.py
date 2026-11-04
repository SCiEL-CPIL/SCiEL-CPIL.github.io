import time

def construct():
    """
    Constructs a rank-98 decomposition for 4x4 * 4x8 matrix multiplication.
    Uses recursive Strassen decomposition.
    Rank 49 for 4x4*4x4.
    Duplicated for the two 4x4 halves of the 4x8 matrix.
    """
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin

    n, m, p = 4, 4, 8
    # Shapes:
    # U: (n*m, rank) = (16, rank)
    # V: (m*p, rank) = (32, rank)
    # W: (p*n, rank) = (32, rank)
    
    # We will build two sets of terms:
    # Set 1: Computes C[0:4, 0:4] = A * B[:, 0:4]
    # Set 2: Computes C[0:4, 4:8] = A * B[:, 4:8]
    
    # Helper to get Strassen terms for 2x2 * 2x2
    # Inputs:
    # A_indices: list of 4 indices corresponding to A[0,0], A[0,1], A[1,0], A[1,1]
    # B_indices: list of 4 indices corresponding to B[0,0], B[0,1], B[1,0], B[1,1]
    # Outputs:
    # list of 7 tuples (u_indices, v_indices, w_indices)
    # u_indices: list of (idx, coeff) for U
    # v_indices: list of (idx, coeff) for V
    # w_indices: list of (idx, coeff) for W
    def get_strassen_2x2(A_idx, B_idx, C_idx_base):
        """
        A_idx: [i00, i01, i10, i11] global indices for A submatrix
        B_idx: [j00, j01, j10, j11] global indices for B submatrix
        C_idx_base: not used directly, we return the W indices for the 4 outputs
        Returns 7 terms. Each term is a dict with 'u', 'v', 'w' lists.
        u, v, w entries are (index, coefficient).
        """
        terms = []
        
        # M1 = (A00+A11)(B00+B11)
        # U: A00(1), A11(1)
        # V: B00(1), B11(1)
        # W: C00(1), C11(1)
        terms.append({
            'u': [(A_idx[0], 1), (A_idx[3], 1)],
            'v': [(B_idx[0], 1), (B_idx[3], 1)],
            'w': [(C_idx_base[0], 1), (C_idx_base[3], 1)]
        })
        
        # M2 = (A10+A11)B00
        # U: A10(1), A11(1)
        # V: B00(1)
        # W: C10(1), C11(-1)
        terms.append({
            'u': [(A_idx[2], 1), (A_idx[3], 1)],
            'v': [(B_idx[0], 1)],
            'w': [(C_idx_base[2], 1), (C_idx_base[3], -1)]
        })
        
        # M3 = A00(B01-B11)
        # U: A00(1)
        # V: B01(1), B11(-1)
        # W: C01(1), C11(1)
        terms.append({
            'u': [(A_idx[0], 1)],
            'v': [(B_idx[1], 1), (B_idx[3], -1)],
            'w': [(C_idx_base[1], 1), (C_idx_base[3], 1)]
        })
        
        # M4 = A11(B10-B00)
        # U: A11(1)
        # V: B10(1), B00(-1)
        # W: C00(1), C10(1)
        terms.append({
            'u': [(A_idx[3], 1)],
            'v': [(B_idx[2], 1), (B_idx[0], -1)],
            'w': [(C_idx_base[0], 1), (C_idx_base[2], 1)]
        })
        
        # M5 = (A00+A10)B11
        # U: A00(1), A10(1)
        # V: B11(1)
        # W: C00(-1), C01(1)
        terms.append({
            'u': [(A_idx[0], 1), (A_idx[2], 1)],
            'v': [(B_idx[3], 1)],
            'w': [(C_idx_base[0], -1), (C_idx_base[1], 1)]
        })
        
        # M6 = (A11-A00)(B00+B01)
        # U: A11(1), A00(-1)
        # V: B00(1), B01(1)
        # W: C11(1)
        terms.append({
            'u': [(A_idx[3], 1), (A_idx[0], -1)],
            'v': [(B_idx[0], 1), (B_idx[1], 1)],
            'w': [(C_idx_base[3], 1)]
        })
        
        # M7 = (A01-A11)(B10+B11)
        # U: A01(1), A11(-1)
        # V: B10(1), B11(1)
        # W: C00(1)
        terms.append({
            'u': [(A_idx[1], 1), (A_idx[3], -1)],
            'v': [(B_idx[2], 1), (B_idx[3], 1)],
            'w': [(C_idx_base[0], 1)]
        })
        
        return terms

    # Generate 49 terms for 4x4 * 4x4
    # A is 4x4, B is 4x4.
    # We view them as 2x2 blocks of 2x2 matrices.
    # A_blocks:
    # A00: rows 0,1 cols 0,1 -> Indices 0,1,4,5
    # A01: rows 0,1 cols 2,3 -> Indices 2,3,6,7
    # A10: rows 2,3 cols 0,1 -> Indices 8,9,12,13
    # A11: rows 2,3 cols 2,3 -> Indices 10,11,14,15
    
    # B_blocks:
    # B00: rows 0,1 cols 0,1 -> Indices 0,1,4,5 (if B is 4x4)
    # B01: rows 0,1 cols 2,3 -> Indices 2,3,6,7
    # B10: rows 2,3 cols 0,1 -> Indices 8,9,12,13
    # B11: rows 2,3 cols 2,3 -> Indices 10,11,14,15
    
    # C_blocks:
    # C00: rows 0,1 cols 0,1 -> Indices 0,1,4,5 (if C is 4x4, col-major k*n+i)
    # C01: rows 0,1 cols 2,3 -> Indices 2,3,6,7
    # C10: rows 2,3 cols 0,1 -> Indices 8,9,12,13
    # C11: rows 2,3 cols 2,3 -> Indices 10,11,14,15
    
    # Note on C index convention: W[k*n+i, r].
    # For C (4x4):
    # C[0,0] (k=0, i=0) -> 0
    # C[0,1] (k=0, i=1) -> 1
    # C[1,0] (k=1, i=0) -> 4
    # C[1,1] (k=1, i=1) -> 5
    # C[0,2] (k=2, i=0) -> 8
    # C[0,3] (k=3, i=0) -> 12
    # etc.
    
    # So for C sub-blocks:
    # C00 (top-left 2x2): indices 0, 1, 4, 5. Order: C[0,0], C[0,1], C[1,0], C[1,1].
    # C01 (top-right 2x2): indices 8, 9, 12, 13. Order: C[2,0], C[2,1], C[3,0], C[3,1].
    #    Wait, C[0,2] is k=2, i=0 -> 2*4+0 = 8. C[0,3] is k=3, i=0 -> 12.
    #    So C01 block corresponds to cols 2,3.
    #    Indices: k=2,3; i=0,1.
    #    C[2,0]=8, C[2,1]=9, C[3,0]=12, C[3,1]=13.
    #    In get_strassen_2x2, C_idx is [C00, C01, C10, C11] of the 2x2 block.
    #    For block C01 (global cols 2,3, rows 0,1):
    #    Local 0,0 -> Global C[0,2] -> 8
    #    Local 0,1 -> Global C[0,3] -> 12
    #    Local 1,0 -> Global C[1,2] -> 2*4+1 = 9
    #    Local 1,1 -> Global C[1,3] -> 3*4+1 = 13
    #    So C_idx_block_01 = [8, 12, 9, 13].
    
    # Let's define the index mappings carefully.
    
    def get_block_indices(matrix_type, block_r, block_c):
        """
        matrix_type: 'A', 'B', 'C'
        block_r: 0 or 1 (block row)
        block_c: 0 or 1 (block col)
        Returns list of 4 global indices [local_00, local_01, local_10, local_11]
        """
        if matrix_type == 'A':
            # A is 4x4. Index = i*4 + j.
            # Block (br, bc) covers rows br*2, br*2+1 and cols bc*2, bc*2+1.
            r0 = block_r * 2
            c0 = block_c * 2
            return [
                r0 * 4 + c0,       # local 0,0
                r0 * 4 + c0 + 1,   # local 0,1
                (r0+1) * 4 + c0,   # local 1,0
                (r0+1) * 4 + c0 + 1 # local 1,1
            ]
        elif matrix_type == 'B':
            # B is 4x4 (conceptually for one half). Index = j*4 + k.
            # Note: In the full problem B is 4x8.
            # For the first half (cols 0-3), k is 0-3. Index = j*4 + k.
            # For the second half (cols 4-7), k is 4-7. Index = j*4 + k.
            # The function get_strassen_2x2 takes B indices directly.
            # We need to pass the correct global B indices.
            # If this is for B1 (cols 0-3):
            #   Block (br, bc) covers rows br*2, br*2+1 and cols bc*2, bc*2+1.
            #   Global col index = col_in_block.
            #   Global index = row * 4 + col.
            # If this is for B2 (cols 4-7):
            #   Global col index = 4 + col_in_block.
            #   Global index = row * 4 + (4 + col_in_block).
            # We will handle the offset outside.
            r0 = block_r * 2
            c0 = block_c * 2
            # Assuming this is for the first half (cols 0-3)
            return [
                r0 * 4 + c0,
                r0 * 4 + c0 + 1,
                (r0+1) * 4 + c0,
                (r0+1) * 4 + c0 + 1
            ]
        elif matrix_type == 'C':
            # C is 4x4 (conceptually for one half). Index = k*4 + i.
            # Block (br, bc) covers rows i in br*2..br*2+1 and cols k in bc*2..bc*2+1.
            r0 = block_r * 2
            c0 = block_c * 2
            return [
                c0 * 4 + r0,       # local 0,0 (k=c0, i=r0)
                (c0+1) * 4 + r0,   # local 0,1 (k=c0+1, i=r0)
                c0 * 4 + (r0+1),   # local 1,0 (k=c0, i=r0+1)
                (c0+1) * 4 + (r0+1) # local 1,1 (k=c0+1, i=r0+1)
            ]
        return []

    # Define the 7 block terms for 4x4 * 4x4
    # Each term is (A_block_indices, B_block_indices, C_block_indices)
    # A_block_indices is a 4x4 matrix of indices? No, it's a linear combination of 4 2x2 blocks.
    # We need to express X_t and Y_t as linear combinations of the 4 sub-blocks of A and B.
    
    # Let A_blocks be A00, A01, A10, A11.
    # Let B_blocks be B00, B01, B10, B11.
    
    # Term 1: M1 = (A00+A11)(B00+B11)
    # X1 = A00 + A11. Y1 = B00 + B11.
    # We need to decompose X1*Y1.
    # X1 is a 2x2 matrix. Its entries are sums of entries from A00 and A11.
    # Specifically, X1[0,0] = A00[0,0] + A11[0,0].
    # This means for the U matrix, we need to add the indices for A00 and A11.
    
    # Let's generate the 7 scalar terms for each of the 7 block terms.
    
    def generate_4x4_terms(b_offset):
        """
        Generates 49 terms for 4x4 * 4x4 multiplication.
        b_offset: 0 for B1 (cols 0-3), 4 for B2 (cols 4-7).
        Returns list of 49 dicts: {'u': [...], 'v': [...], 'w': [...]}
        """
        terms = []
        
        # Indices for A blocks (constant)
        A_00 = get_block_indices('A', 0, 0)
        A_01 = get_block_indices('A', 0, 1)
        A_10 = get_block_indices('A', 1, 0)
        A_11 = get_block_indices('A', 1, 1)
        
        # Indices for B blocks (depends on offset)
        B_00_base = get_block_indices('B', 0, 0)
        B_01_base = get_block_indices('B', 0, 1)
        B_10_base = get_block_indices('B', 1, 0)
        B_11_base = get_block_indices('B', 1, 1)
        
        # Apply offset to B indices
        B_00 = [idx + b_offset for idx in B_00_base]
        B_01 = [idx + b_offset for idx in B_01_base]
        B_10 = [idx + b_offset for idx in B_10_base]
        B_11 = [idx + b_offset for idx in B_11_base]
        
        # Indices for C blocks (constant for C1, but we are generating for one 4x4 output block)
        # The output C is 4x4.
        C_00 = get_block_indices('C', 0, 0)
        C_01 = get_block_indices('C', 0, 1)
        C_10 = get_block_indices('C', 1, 0)
        C_11 = get_block_indices('C', 1, 1)
        
        # Define the 7 block terms X_t, Y_t
        # X_t is a linear combo of A_blocks.
        # Y_t is a linear combo of B_blocks.
        # We represent X_t as a list of (A_block, coeff)
        # Y_t as a list of (B_block, coeff)
        
        block_terms = [
            # (X_combos, Y_combos, C_block)
            ([('A00', 1), ('A11', 1)], [('B00', 1), ('B11', 1)], C_00), # M1
            ([('A10', 1), ('A11', 1)], [('B00', 1)], C_10),              # M2
            ([('A00', 1)], [('B01', 1), ('B11', -1)], C_01),             # M3
            ([('A11', 1)], [('B10', 1), ('B00', -1)], C_00),             # M4
            ([('A00', 1), ('A10', 1)], [('B11', 1)], C_00),              # M5
            ([('A11', 1), ('A00', -1)], [('B00', 1), ('B01', 1)], C_11), # M6
            ([('A01', 1), ('A11', -1)], [('B10', 1), ('B11', 1)], C_00)  # M7
        ]
        
        # Map block names to indices
        A_blocks = {'A00': A_00, 'A01': A_01, 'A10': A_10, 'A11': A_11}
        B_blocks = {'B00': B_00, 'B01': B_01, 'B10': B_10, 'B11': B_11}
        
        for x_combos, y_combos, c_block in block_terms:
            # Construct X and Y as 2x2 matrices of indices with coefficients
            # X[local_r, local_c] = sum of coeffs * A_block[local_r, local_c]
            # We need to pass the actual indices to get_strassen_2x2.
            # But get_strassen_2x2 expects 4 indices for A and 4 for B.
            # Here, the "A" for the 2x2 strassen is the matrix X, which is a sum of blocks.
            # So X[0,0] is a linear combination of A_block[0,0]s.
            # We can just sum the indices with coefficients.
            
            X_indices = [[0, 0], [0, 0]] # Will be replaced
            # Actually, we need to build the linear combination for each of the 4 entries.
            # X_entry = sum( coeff * A_block_entry )
            
            # Let's build the U contributions directly.
            # For a specific 2x2 Strassen term, we need the indices of X and Y.
            # X is not a single index, it's a matrix of linear combinations.
            
            # Alternative approach:
            # The 2x2 Strassen algorithm operates on the entries of X and Y.
            # Let's generate the 7 scalar multiplications for the block product X*Y.
            # We need to know the "index" of each entry of X and Y in the global U/V matrices.
            # But X is not a global matrix entry, it's a virtual matrix.
            
            # We must expand the virtual matrix X into its constituent global A indices.
            
            # Let's create a helper to get the global indices for each entry of X.
            # X[0,0] corresponds to A_block[0,0] for each block in the combo.
            
            # We will build the U, V, W lists for the 7 terms of this block product.
            
            # Get the 7 terms of the 2x2 Strassen on virtual matrices X and Y.
            # We need to pass the "indices" of X and Y to get_strassen_2x2.
            # But get_strassen_2x2 just adds the indices it's given.
            # So we need to provide the list of global indices that make up X[0,0], X[0,1], etc.
            
            # Let's build X_flat and Y_flat as lists of (global_index, coeff) for each of the 4 entries.
            X_entries = [[[], []], [[], []]]
            Y_entries = [[[], []], [[], []]]
            
            for block_name, coeff in x_combos:
                idxs = A_blocks[block_name]
                for local_r in range(2):
                    for local_c in range(2):
                        X_entries[local_r][local_c].append((idxs[local_r * 2 + local_c], coeff))
            
            for block_name, coeff in y_combos:
                idxs = B_blocks[block_name]
                for local_r in range(2):
                    for local_c in range(2):
                        Y_entries[local_r][local_c].append((idxs[local_r * 2 + local_c], coeff))
            
            # Now we have the linear combinations for the 4 entries of X and Y.
            # We need to apply the 2x2 Strassen algorithm.
            # The 2x2 Strassen terms are defined in get_strassen_2x2.
            # We can't just call it with single indices.
            # We need to replicate the logic of get_strassen_2x2 but with lists of (idx, coeff).
            
            # Let's write a specialized function for this.
            
            # Term 1: (X00+X11)(Y00+Y11)
            # U: X00(1), X11(1)
            # V: Y00(1), Y11(1)
            # W: C00(1), C11(1)
            
            def add_terms(target_u, target_v, target_w, x_list, y_list, w_indices, w_coeffs):
                for idx, c in x_list:
                    target_u.append((idx, c))
                for idx, c in y_list:
                    target_v.append((idx, c))
                for i, idx in enumerate(w_indices):
                    target_w.append((idx, w_coeffs[i]))
            
            # M1
            t1_u = []
            t1_v = []
            t1_w = []
            add_terms(t1_u, t1_v, t1_w, X_entries[0][0], Y_entries[0][0], [c_block[0]], [1])
            add_terms(t1_u, t1_v, t1_w, X_entries[1][1], Y_entries[1][1], [c_block[3]], [1])
            terms.append({'u': t1_u, 'v': t1_v, 'w': t1_w})
            
            # M2: (X10+X11)Y00
            t2_u = []
            t2_v = []
            t2_w = []
            add_terms(t2_u, t2_v, t2_w, X_entries[1][0], Y_entries[0][0], [c_block[2]], [1])
            add_terms(t2_u, t2_v, t2_w, X_entries[1][1], Y_entries[0][0], [c_block[3]], [-1])
            terms.append({'u': t2_u, 'v': t2_v, 'w': t2_w})
            
            # M3: X00(Y01-Y11)
            t3_u = []
            t3_v = []
            t3_w = []
            add_terms(t3_u, t3_v, t3_w, X_entries[0][0], Y_entries[0][1], [c_block[1]], [1])
            add_terms(t3_u, t3_v, t3_w, X_entries[0][0], Y_entries[1][1], [c_block[3]], [1])
            # Note: Y11 has coeff -1 in the sum, so we add (idx, -1)
            # Wait, my add_terms takes the list from Y_entries which already has coeffs.
            # Y_entries[1][1] contains (idx, coeff).
            # We need to multiply the coeff by -1? No, the term is X00 * (Y01 - Y11).
            # So U is X00. V is Y01(1) and Y11(-1).
            # My Y_entries[1][1] has the coeffs from the block combination.
            # So if Y11 was (idx, 1), it becomes (idx, 1).
            # We need to negate it for the -Y11 part.
            # Let's handle the sign in the addition.
            
            # Let's redo M3 properly.
            t3_u = []
            t3_v = []
            t3_w = []
            for idx, c in X_entries[0][0]:
                t3_u.append((idx, c))
            for idx, c in Y_entries[0][1]:
                t3_v.append((idx, c))
            for idx, c in Y_entries[1][1]:
                t3_v.append((idx, -c))
            t3_w.append((c_block[1], 1))
            t3_w.append((c_block[3], 1))
            terms.append({'u': t3_u, 'v': t3_v, 'w': t3_w})
            
            # M4: X11(Y10-Y00)
            t4_u = []
            t4_v = []
            t4_w = []
            for idx, c in X_entries[1][1]:
                t4_u.append((idx, c))
            for idx, c in Y_entries[1][0]:
                t4_v.append((idx, c))
            for idx, c in Y_entries[0][0]:
                t4_v.append((idx, -c))
            t4_w.append((c_block[0], 1))
            t4_w.append((c_block[2], 1))
            terms.append({'u': t4_u, 'v': t4_v, 'w': t4_w})
            
            # M5: (X00+X10)Y11
            t5_u = []
            t5_v = []
            t5_w = []
            for idx, c in X_entries[0][0]:
                t5_u.append((idx, c))
            for idx, c in X_entries[1][0]:
                t5_u.append((idx, c))
            for idx, c in Y_entries[1][1]:
                t5_v.append((idx, c))
            t5_w.append((c_block[0], -1))
            t5_w.append((c_block[1], 1))
            terms.append({'u': t5_u, 'v': t5_v, 'w': t5_w})
            
            # M6: (X11-X00)(Y00+Y01)
            t6_u = []
            t6_v = []
            t6_w = []
            for idx, c in X_entries[1][1]:
                t6_u.append((idx, c))
            for idx, c in X_entries[0][0]:
                t6_u.append((idx, -c))
            for idx, c in Y_entries[0][0]:
                t6_v.append((idx, c))
            for idx, c in Y_entries[0][1]:
                t6_v.append((idx, c))
            t6_w.append((c_block[3], 1))
            terms.append({'u': t6_u, 'v': t6_v, 'w': t6_w})
            
            # M7: (X01-X11)(Y10+Y11)
            t7_u = []
            t7_v = []
            t7_w = []
            for idx, c in X_entries[0][1]:
                t7_u.append((idx, c))
            for idx, c in X_entries[1][1]:
                t7_u.append((idx, -c))
            for idx, c in Y_entries[1][0]:
                t7_v.append((idx, c))
            for idx, c in Y_entries[1][1]:
                t7_v.append((idx, c))
            t7_w.append((c_block[0], 1))
            terms.append({'u': t7_u, 'v': t7_v, 'w': t7_w})
            
        return terms

    # Generate terms for first half (B1, C1)
    terms1 = generate_4x4_terms(0)
    
    # Generate terms for second half (B2, C2)
    # B2 indices are B1 indices + 4.
    # C2 indices are C1 indices + ?
    # C1 cols 0-3. C2 cols 4-7.
    # C index = k*4 + i.
    # For C1, k in 0-3.
    # For C2, k in 4-7.
    # So C2 index = (k+4)*4 + i = k*4 + i + 16.
    # So we can just offset C indices by 16.
    
    # My generate_4x4_terms function hardcodes C indices via get_block_indices.
    # I need to modify it to accept a C offset or regenerate with offset.
    # Let's just modify the function to accept offsets.
    
    def generate_4x4_terms_v2(b_offset, c_offset):
        # Same as before but with c_offset
        # ... (copying logic) ...
        # To save time, I'll assume I can just add the offset to the W indices in the output.
        # The W indices in the terms are global C indices.
        # For C2, we add 16 to all W indices.
        pass

    # Let's just reuse generate_4x4_terms and post-process W indices.
    # The U and V indices for B2 need offset 4.
    # The U indices for A are the same.
    # The W indices for C2 need offset 16.
    
    terms2 = generate_4x4_terms(4)
    
    # Adjust W indices in terms2 by adding 16
    for t in terms2:
        t['w'] = [(idx + 16, coeff) for idx, coeff in t['w']]
        
    all_terms = terms1 + terms2
    rank = len(all_terms)
    
    U = [[0] * rank for _ in range(16)]
    V = [[0] * rank for _ in range(32)]
    W = [[0] * rank for _ in range(32)]
    
    for r, term in enumerate(all_terms):
        for idx, coeff in term['u']:
            if 0 <= idx < 16:
                U[idx][r] += coeff
        for idx, coeff in term['v']:
            if 0 <= idx < 32:
                V[idx][r] += coeff
        for idx, coeff in term['w']:
            if 0 <= idx < 32:
                W[idx][r] += coeff
                
    return {"U": U, "V": V, "W": W}
