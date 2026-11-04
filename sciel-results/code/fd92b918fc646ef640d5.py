"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # n=4, m=4, p=6
    
    # Helper to get 2x2x2 Strassen decomposition
    # Returns U (4x7), V (4x7), W (4x7)
    # U index: i*2+j
    # V index: j*2+k
    # W index: k*2+i
    def get_strassen_2x2x2():
        U = [[0]*7 for _ in range(4)]
        V = [[0]*7 for _ in range(4)]
        W = [[0]*7 for _ in range(4)]
        
        # M1: (A00+A11)(B00+B11)
        # A00: U[0], A11: U[3]
        # B00: V[0], B11: V[3]
        U[0][0] = 1; U[3][0] = 1
        V[0][0] = 1; V[3][0] = 1
        # C00 += M1, C11 += M1
        W[0][0] = 1; W[3][0] = 1 # C00 (k=0,i=0), C11 (k=1,i=1)
        
        # M2: (A10+A11)B00
        # A10: U[2], A11: U[3]
        # B00: V[0]
        U[2][1] = 1; U[3][1] = 1
        V[0][1] = 1
        # C10 += M2, C11 -= M2
        W[2][1] = 1; W[3][1] = -1 # C10 (k=0,i=1), C11 (k=1,i=1)
        
        # M3: A00(B01-B11)
        # A00: U[0]
        # B01: V[1], B11: V[3]
        U[0][2] = 1
        V[1][2] = 1; V[3][2] = -1
        # C01 += M3, C11 += M3
        W[1][2] = 1; W[3][2] = 1 # C01 (k=1,i=0), C11 (k=1,i=1)
        
        # M4: A11(B10-B00)
        # A11: U[3]
        # B10: V[2], B00: V[0]
        U[3][3] = 1
        V[2][3] = 1; V[0][3] = -1
        # C00 += M4, C10 += M4
        W[0][3] = 1; W[2][3] = 1 # C00 (k=0,i=0), C10 (k=0,i=1)
        
        # M5: (A00+A01)B11
        # A00: U[0], A01: U[1]
        # B11: V[3]
        U[0][4] = 1; U[1][4] = 1
        V[3][4] = 1
        # C00 -= M5, C01 += M5
        W[0][4] = -1; W[1][4] = 1 # C00 (k=0,i=0), C01 (k=1,i=0)
        
        # M6: (A10-A00)(B00+B01)
        # A10: U[2], A00: U[0]
        # B00: V[0], B01: V[1]
        U[2][5] = 1; U[0][5] = -1
        V[0][5] = 1; V[1][5] = 1
        # C11 += M6
        W[3][5] = 1 # C11 (k=1,i=1)
        
        # M7: (A01-A11)(B10+B11)
        # A01: U[1], A11: U[3]
        # B10: V[2], B11: V[3]
        U[1][6] = 1; U[3][6] = -1
        V[2][6] = 1; V[3][6] = 1
        # C00 += M7, C11 += M7
        W[0][6] = 1; W[3][6] = 1 # C00 (k=0,i=0), C11 (k=1,i=1)
        
        return U, V, W

    U_full = []
    V_full = []
    W_full = []
    
    # Part 1: 4x4 x 4x4
    # We use 7 Strassen steps.
    # Each step is a 2x2x2 multiplication of blocks.
    # We need to map the 2x2 block indices to 4x4 indices.
    
    # Block indices for A (4x4):
    # A00: rows 0-1, cols 0-1
    # A01: rows 0-1, cols 2-3
    # A10: rows 2-3, cols 0-1
    # A11: rows 2-3, cols 2-3
    
    # Block indices for B (4x4 part):
    # B00: rows 0-1, cols 0-1
    # B01: rows 0-1, cols 2-3
    # B10: rows 2-3, cols 0-1
    # B11: rows 2-3, cols 2-3
    
    # Block indices for C (4x4 part):
    # C00: rows 0-1, cols 0-1
    # C01: rows 0-1, cols 2-3
    # C10: rows 2-3, cols 0-1
    # C11: rows 2-3, cols 2-3
    
    # We will add 7 * 7 = 49 columns.
    
    def add_block_decomp(U_block, V_block, W_block, row_offset_U, col_offset_U, 
                         row_offset_V, col_offset_V, 
                         row_offset_W, col_offset_W, scale=1):
        # This function is complex to generalize directly with offsets for the 4x4 case
        # because the Strassen formula mixes blocks.
        # It is easier to construct the 49 columns explicitly.
        pass

    # Explicit construction for 4x4x4 Strassen
    # We reuse the 2x2x2 logic but at the block level.
    # The "elements" of the 2x2 matrices are 2x2 blocks.
    # So for M1 = (A00+A11)(B00+B11), we need to decompose the product of two 2x2 matrices.
    # This product is a 2x2 matrix.
    # We use the 2x2x2 Strassen decomposition for this product.
    # The U, V, W of the 2x2x2 decomposition operate on the "elements" of the blocks.
    
    # Let's create a helper to expand a 2x2x2 decomp into the full 4x4 indices.
    def expand_2x2x2_to_4x4x4(U2, V2, W2, 
                               A_block_map, B_block_map, C_block_map):
        # A_block_map: dict from 2x2 index (0..3) to list of 4x4 indices (row, col)
        # Actually, the 2x2x2 decomp U has 4 rows.
        # Row 0 corresponds to A00, Row 1 to A01, etc.
        # We need to map these to the actual 4x4 U indices.
        
        pass

    # This approach is getting complicated. 
    # Alternative: Just generate the 49 columns by iterating through the 7 Strassen steps
    # and for each step, iterating through the 7 sub-steps of the 2x2x2 decomp.
    
    # Strassen Steps for 4x4x4:
    # 1. M1 = (A00+A11)(B00+B11)
    # 2. M2 = (A10+A11)B00
    # 3. M3 = A00(B01-B11)
    # 4. M4 = A11(B10-B00)
    # 5. M5 = (A00+A01)B11
    # 6. M6 = (A10-A00)(B00+B01)
    # 7. M7 = (A01-A11)(B10+B11)
    
    # For each M, we have a 2x2x2 multiplication.
    # We use the U, V, W from get_strassen_2x2x2().
    
    # We need to define the mapping from 2x2 block indices to 4x4 indices.
    # A blocks:
    # A00: rows [0,1], cols [0,1] -> U indices: 0,1,4,5 (row*4+col)
    # A01: rows [0,1], cols [2,3] -> U indices: 2,3,6,7
    # A10: rows [2,3], cols [0,1] -> U indices: 8,9,12,13
    # A11: rows [2,3], cols [2,3] -> U indices: 10,11,14,15
    
    # B blocks (for 4x4 part):
    # B00: rows [0,1], cols [0,1] -> V indices: 0,1,4,5
    # B01: rows [0,1], cols [2,3] -> V indices: 2,3,6,7
    # B10: rows [2,3], cols [0,1] -> V indices: 8,9,12,13
    # B11: rows [2,3], cols [2,3] -> V indices: 10,11,14,15
    
    # C blocks (for 4x4 part):
    # C00: rows [0,1], cols [0,1] -> W indices: 0,1,4,5 (col*4+row)
    # C01: rows [0,1], cols [2,3] -> W indices: 8,9,12,13
    # C10: rows [2,3], cols [0,1] -> W indices: 2,3,6,7
    # C11: rows [2,3], cols [2,3] -> W indices: 10,11,14,15
    
    # The 2x2x2 decomp U has 4 rows.
    # Row 0: A00
    # Row 1: A01
    # Row 2: A10
    # Row 3: A11
    
    # We need to map these 4 rows to the 16 rows of the 4x4 U.
    # For a given Strassen step, the "A" input is a combination of blocks.
    # e.g. M1: A_part = A00 + A11.
    # The 2x2x2 decomp expects A00, A01, A10, A11 as separate inputs.
    # But here, the "A00" of the sub-problem is the block A00 of the main problem.
    # Wait, the sub-problem is multiplying two 2x2 matrices.
    # Let X be the first matrix, Y be the second.
    # X = A00+A11. Y = B00+B11.
    # X is a 2x2 matrix. Its elements are the elements of A00+A11.
    # So X[0,0] = A[0,0] + A[2,2].
    # The 2x2x2 decomp U[0] corresponds to X[0,0].
    # So U_full[row_for_A00_00][col] += U2[0][sub_col]
    # U_full[row_for_A11_00][col] += U2[0][sub_col]
    
    # This is the key.
    
    U2, V2, W2 = get_strassen_2x2x2()
    
    # Define block index mappings
    # Block (bi, bj) -> list of 4x4 indices
    # A: row*4+col
    # B: row*4+col
    # C: col*4+row
    
    def get_A_indices(bi, bj):
        # bi, bj in {0,1}
        r0 = bi*2; r1 = bi*2+1
        c0 = bj*2; c1 = bj*2+1
        return [r0*4+c0, r0*4+c1, r1*4+c0, r1*4+c1]
        
    def get_B_indices(bi, bj):
        r0 = bi*2; r1 = bi*2+1
        c0 = bj*2; c1 = bj*2+1
        return [r0*4+c0, r0*4+c1, r1*4+c0, r1*4+c1]
        
    def get_C_indices(bi, bj):
        # C block (bi, bj) corresponds to rows bi*2..bi*2+1, cols bj*2..bj*2+1
        # W index: col*4+row
        r0 = bi*2; r1 = bi*2+1
        c0 = bj*2; c1 = bj*2+1
        return [c0*4+r0, c0*4+r1, c1*4+r0, c1*4+r1]

    # Strassen Steps
    # Each step has:
    # 1. A blocks involved and coefficients
    # 2. B blocks involved and coefficients
    # 3. C blocks involved and coefficients (from the 2x2x2 W2)
    
    # Step 1: M1 = (A00+A11)(B00+B11)
    # A: 00 (1), 11 (1)
    # B: 00 (1), 11 (1)
    # C: from W2
    
    steps = [
        # (A_blocks, B_blocks)
        # A_blocks: list of (block_idx, coeff)
        # block_idx: (bi, bj)
        [
            { 'A': [((0,0), 1), ((1,1), 1)], 'B': [((0,0), 1), ((1,1), 1)] },
            { 'A': [((1,0), 1), ((1,1), 1)], 'B': [((0,0), 1)] },
            { 'A': [((0,0), 1)], 'B': [((0,1), 1), ((1,1), -1)] },
            { 'A': [((1,1), 1)], 'B': [((1,0), 1), ((0,0), -1)] },
            { 'A': [((0,0), 1), ((0,1), 1)], 'B': [((1,1), 1)] },
            { 'A': [((1,0), 1), ((0,0), -1)], 'B': [((0,0), 1), ((0,1), 1)] },
            { 'A': [((0,1), 1), ((1,1), -1)], 'B': [((1,0), 1), ((1,1), 1)] },
        ]
    ]
    
    rank_444 = 0
    for step in steps[0]:
        # For each sub-column in U2 (0..6)
        for sub_c in range(7):
            col_idx = rank_444 + sub_c
            
            # U
            for (bi, bj), coeff in step['A']:
                if coeff != 0:
                    # The 2x2x2 U2 row corresponding to this block?
                    # No, U2 has 4 rows for the 4 elements of the 2x2 matrix.
                    # The "A" in the step is a 2x2 matrix formed by blocks.
                    # The elements of this 2x2 matrix are the elements of the blocks.
                    # So U2[0] corresponds to element (0,0) of the A-matrix.
                    # Element (0,0) of A-matrix is the sum of A_block(0,0)[0,0] and A_block(1,1)[0,0]?
                    # No.
                    # The Strassen step M1 = (A00+A11)(B00+B11).
                    # Let X = A00+A11. X is 2x2.
                    # X[0,0] = A00[0,0] + A11[0,0].
                    # The 2x2x2 decomp U2[0] multiplies X[0,0].
                    # So U_full[A00_idx[0]] += U2[0][sub_c]
                    # U_full[A11_idx[0]] += U2[0][sub_c]
                    
                    # Which row of U2 corresponds to which element?
                    # U2 row 0: X[0,0]
                    # U2 row 1: X[0,1]
                    # U2 row 2: X[1,0]
                    # U2 row 3: X[1,1]
                    
                    # The blocks in step['A'] are the blocks that form X.
                    # But X is a 2x2 matrix of scalars? No, X is a 2x2 matrix of blocks?
                    # No, A00 is a 2x2 block. A is 4x4.
                    # X = A00 + A11 is a 2x2 matrix? No, A00 is 2x2.
                    # So X is 2x2.
                    # The elements of X are scalars.
                    # X[0,0] = A[0,0] + A[2,2].
                    # So we add U2[0][sub_c] to U_full[A[0,0]] and U_full[A[2,2]].
                    
                    # But step['A'] gives blocks.
                    # We need to map the 4 elements of X to the 4 elements of the blocks.
                    # X[0,0] comes from A_block(0,0)[0,0] and A_block(1,1)[0,0].
                    # X[0,1] comes from A_block(0,0)[0,1] and A_block(1,1)[0,1].
                    # etc.
                    
                    # So for each block (bi,bj) with coeff:
                    # for local_r in 0..1:
                    #   for local_c in 0..1:
                    #     global_r = bi*2 + local_r
                    #     global_c = bj*2 + local_c
                    #     U_full[global_r*4+global_c] += coeff * U2[local_r*2+local_c][sub_c]
                    
                    for lr in range(2):
                        for lc in range(2):
                            gr = bi*2 + lr
                            gc = bj*2 + lc
                            u_idx = gr*4 + gc
                            u2_val = U2[lr*2+lc][sub_c]
                            if u2_val != 0:
                                # Add to U_full
                                pass # Need to store U_full

    # This logic is correct but needs to be implemented in the loop.
    # I will implement it directly in the code.
    
    # Part 2: 4x4 x 4x2
    # Rank 28.
    # 4 blocks of 2x2x2.
    # C2 = A11 Btop + A12 Bbot (Top 2 rows of C2)
    # C2 = A21 Btop + A22 Bbot (Bot 2 rows of C2)
    # Btop: rows 0-1 of B2 (cols 4-5)
    # Bbot: rows 2-3 of B2 (cols 4-5)
    
    # We will add 28 columns.
    
    return {"U": U_full, "V": V_full, "W": W_full}

# EVOLVE-BLOCK-END
