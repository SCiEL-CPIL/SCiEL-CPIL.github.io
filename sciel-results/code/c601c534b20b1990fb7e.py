import numpy as np

DIMENSIONS = [4, 4, 7]

def construct():
    n, m, p = DIMENSIONS
    # We will construct a decomposition of rank 98.
    # Part 1: 4x4x4 multiplication using Strassen (rank 49)
    # Part 2: 4x4x3 multiplication using Strassen on 4x4x4 with last col zeroed (rank 49)
    
    # 2x2x2 Strassen decomposition
    # U, V, W are 4x7
    U22 = np.zeros((4, 7))
    V22 = np.zeros((4, 7))
    W22 = np.zeros((4, 7))
    
    # r=0: S1 = (A00+A11)(B00+B11)
    U22[0, 0] = 1; U22[3, 0] = 1
    V22[0, 0] = 1; V22[3, 0] = 1
    W22[0, 0] = 1; W22[3, 0] = 1
    
    # r=1: S2 = (A10+A11)B00
    U22[2, 1] = 1; U22[3, 1] = 1
    V22[0, 1] = 1
    W22[1, 1] = 1; W22[3, 1] = -1
    
    # r=2: S3 = A00(B01-B11)
    U22[0, 2] = 1
    V22[1, 2] = 1; V22[3, 2] = -1
    W22[1, 2] = 1; W22[3, 2] = 1
    
    # r=3: S4 = A11(B10-B00)
    U22[3, 3] = 1
    V22[2, 3] = 1; V22[0, 3] = -1
    W22[0, 3] = 1; W22[1, 3] = 1
    
    # r=4: S5 = (A00+A01)B11
    U22[0, 4] = 1; U22[1, 4] = 1
    V22[3, 4] = 1
    W22[0, 4] = -1; W22[1, 4] = 1
    
    # r=5: S6 = (A10-A00)(B01+B11)
    U22[2, 5] = 1; U22[0, 5] = -1
    V22[1, 5] = 1; V22[3, 5] = 1
    W22[3, 5] = 1
    
    # r=6: S7 = (A01-A11)(B00+B10)
    U22[1, 6] = 1; U22[3, 6] = -1
    V22[0, 6] = 1; V22[2, 6] = 1
    W22[0, 6] = 1; W22[3, 6] = 1
    
    # Now build 4x4x4
    U44 = np.zeros((16, 49))
    V44 = np.zeros((16, 49))
    W44 = np.zeros((16, 49))
    
    # Strassen steps for 4x4x4
    # M1 = (A10+A00)(B00+B11)
    # M2 = (A10+A00)B11
    # M3 = A00(B01-B11)
    # M4 = A11(B10-B00)
    # M5 = (A00+A11)(B00+B10)
    # M6 = (A10-A00)(B01+B11)
    # M7 = (A01-A11)(B00+B10)
    
    # Define block mappings
    # A blocks: 00, 01, 10, 11
    # B blocks: 00, 01, 10, 11
    # C blocks: 00, 01, 10, 11
    
    # For each Strassen term, we map the 2x2 decomposition to the 4x4 blocks
    # Term 1: A' = A10+A00, B' = B00+B11
    # A10 is rows 2-3, cols 0-1. A00 is rows 0-1, cols 0-1.
    # In U22, row 0 is A00 (top-left), row 1 is A01 (top-right), row 2 is A10 (bottom-left), row 3 is A11 (bottom-right).
    # So for A10+A00, we map U22 rows 0,1,2,3 to 4x4 rows:
    # U22[0] (A00) -> 4x4 A00 (rows 0-1, cols 0-1) -> U44 indices 0,1,4,5? No.
    # U index is i*m+j.
    # A00: i=0,1; j=0,1. Indices: 0,1,4,5.
    # A01: i=0,1; j=2,3. Indices: 2,3,6,7.
    # A10: i=2,3; j=0,1. Indices: 8,9,12,13.
    # A11: i=2,3; j=2,3. Indices: 10,11,14,15.
    
    # We need a mapping from 2x2 block index to 4x4 indices.
    # Block (bi, bj) where bi in 0,1, bj in 0,1.
    # Rows: bi*2, bi*2+1. Cols: bj*2, bj*2+1.
    # U indices:
    # (0,0) -> 0, 1, 4, 5
    # (0,1) -> 2, 3, 6, 7
    # (1,0) -> 8, 9, 12, 13
    # (1,1) -> 10, 11, 14, 15
    
    # V indices (j*p+k, p=4):
    # Block (bj, bk)
    # (0,0) -> 0, 1, 4, 5
    # (0,1) -> 2, 3, 6, 7
    # (1,0) -> 8, 9, 12, 13
    # (1,1) -> 10, 11, 14, 15
    
    # W indices (k*n+i, n=4):
    # Block (bk, bi)
    # (0,0) -> 0, 1, 4, 5
    # (0,1) -> 2, 3, 6, 7
    # (1,0) -> 8, 9, 12, 13
    # (1,1) -> 10, 11, 14, 15
    
    def get_indices(block_row, block_col, dim):
        # block_row, block_col in 0,1
        # dim is 4 (size of block is 2)
        start_r = block_row * 2
        start_c = block_col * 2
        idxs = []
        for dr in range(2):
            for dc in range(2):
                r = start_r + dr
                c = start_c + dc
                idxs.append(r * dim + c)
        return idxs
        
    # We will add the 7 sub-terms for each of the 7 Strassen terms.
    # Total 49 terms.
    
    # Strassen Term Definitions:
    # (A_blocks, B_blocks, C_blocks, sign)
    # A_blocks: list of (block_idx, sign)
    # B_blocks: list of (block_idx, sign)
    # C_blocks: list of (block_idx, sign)
    
    # M1 = (A10+A00)(B00+B11)
    # C00 += M1, C11 += M1
    # M2 = (A10+A00)B11
    # C10 += M2, C11 -= M2
    # M3 = A00(B01-B11)
    # C01 += M3, C11 += M3
    # M4 = A11(B10-B00)
    # C00 += M4, C10 += M4
    # M5 = (A00+A11)(B00+B10)
    # C00 -= M5, C01 += M5
    # M6 = (A10-A00)(B01+B11)
    # C11 += M6
    # M7 = (A01-A11)(B00+B10)
    # C00 += M7, C11 += M7
    
    # Note: The C blocks are determined by the Strassen formula.
    # We iterate through the 7 Strassen terms.
    # For each term, we iterate through the 7 sub-terms of the 2x2 decomposition.
    
    strassen_terms = [
        # (A_blocks, B_blocks, C_blocks)
        # A_blocks: list of (block, coeff)
        # block is (row, col)
        ([((1,0), 1), ((0,0), 1)], [((0,0), 1), ((1,1), 1)], [((0,0), 1), ((1,1), 1)]), # M1
        ([((1,0), 1), ((0,0), 1)], [((1,1), 1)], [((1,0), 1), ((1,1), -1)]), # M2
        ([((0,0), 1)], [((0,1), 1), ((1,1), -1)], [((0,1), 1), ((1,1), 1)]), # M3
        ([((1,1), 1)], [((1,0), 1), ((0,0), -1)], [((0,0), 1), ((1,0), 1)]), # M4
        ([((0,0), 1), ((1,1), 1)], [((0,0), 1), ((1,0), 1)], [((0,0), -1), ((0,1), 1)]), # M5
        ([((1,0), 1), ((0,0), -1)], [((0,1), 1), ((1,1), 1)], [((1,1), 1)]), # M6
        ([((0,1), 1), ((1,1), -1)], [((0,0), 1), ((1,0), 1)], [((0,0), 1), ((1,1), 1)]) # M7
    ]
    
    term_idx = 0
    for A_def, B_def, C_def in strassen_terms:
        for r in range(7):
            # u_val, v_val, w_val from U22, V22, W22
            # We need to map the 2x2 entries to the 4x4 blocks
            # U22[r, c] is non-zero for some c? No, U22 is 4x7.
            # For a fixed r, U22[:, r] has 4 entries.
            # These correspond to the 4 blocks of A'.
            # We need to expand each block entry to the 4 indices in 4x4.
            
            for c in range(4):
                u_val = U22[c, r]
                if u_val == 0: continue
                # c is the block index in 2x2 (0:00, 1:01, 2:10, 3:11)
                # Wait, U22 rows are 0,1,2,3.
                # 0: A00, 1: A01, 2: A10, 3: A11.
                block_row = c // 2
                block_col = c % 2
                # Find which A_def block this is
                # A_def is a list of (block, coeff)
                # We need to check if (block_row, block_col) is in A_def
                # Actually, U22 already encodes the linear combination.
                # So we just map the block (block_row, block_col) to the 4x4 indices.
                # But wait, the Strassen term defines A' as a combination of blocks.
                # U22[c, r] is the coefficient of that block in the r-th sub-term of the 2x2 product.
                # So we just add u_val to the corresponding 4x4 indices.
                
                # Get 4x4 indices for this block
                # Block (block_row, block_col) in A
                idxs = get_indices(block_row, block_col, 4)
                for idx in idxs:
                    U44[idx, term_idx] += u_val
                    
            for c in range(4):
                v_val = V22[c, r]
                if v_val == 0: continue
                block_row = c // 2
                block_col = c % 2
                idxs = get_indices(block_row, block_col, 4)
                for idx in idxs:
                    V44[idx, term_idx] += v_val
                    
            for c in range(4):
                w_val = W22[c, r]
                if w_val == 0: continue
                # W22 rows correspond to C blocks?
                # W index is k*n+i.
                # In 2x2, W index is k*2+i.
                # Row 0: k=0, i=0 -> C00
                # Row 1: k=0, i=1 -> C01? No.
                # W index in 2x2: k*2+i.
                # k=0, i=0 -> 0.
                # k=0, i=1 -> 1.
                # k=1, i=0 -> 2.
                # k=1, i=1 -> 3.
                # So row 0 is C00 (k=0, i=0), row 1 is C01 (k=0, i=1)?
                # No, C is 2x2. C_{ik}.
                # W index is k*n+i.
                # For 2x2, n=2.
                # k=0, i=0 -> 0.
                # k=0, i=1 -> 1.
                # k=1, i=0 -> 2.
                # k=1, i=1 -> 3.
                # So row 0: C_{00}. Row 1: C_{01}. Row 2: C_{10}. Row 3: C_{11}.
                # Block (k_block, i_block).
                # k_block = c // 2, i_block = c % 2.
                # We need to map to C blocks.
                # C blocks are indexed by (k_block, i_block).
                # In C_def, we have blocks (block_row, block_col) which are (i_block, k_block)?
                # C is i x k.
                # Strassen C_def uses (i_block, k_block).
                # So we need to find the coefficient in C_def for (i_block, k_block).
                # But W22 already has the values.
                # We just map the block to 4x4 indices.
                # Block (k_block, i_block) in W22.
                # i_block = c % 2, k_block = c // 2.
                # In 4x4, C block (i_block, k_block).
                # Indices for C block (i_block, k_block) in W (k*n+i):
                # k = k_block*2, k_block*2+1.
                # i = i_block*2, i_block*2+1.
                # W index = k*4 + i.
                
                k_block = c // 2
                i_block = c % 2
                start_k = k_block * 2
                start_i = i_block * 2
                idxs = []
                for dk in range(2):
                    for di in range(2):
                        k = start_k + dk
                        i = start_i + di
                        idxs.append(k * 4 + i)
                for idx in idxs:
                    W44[idx, term_idx] += w_val
                    
            term_idx += 1
            
    # Now we have U44, V44, W44 for 4x4x4.
    # For 4x4x3, we use the same U and W, but modify V.
    # V44 is 16x49.
    # We create V44_3 by zeroing out the entries corresponding to k=3.
    # V index is j*4 + k.
    # k=3 means index % 4 == 3.
    V44_3 = V44.copy()
    for j in range(4):
        for k in range(4):
            if k == 3:
                V44_3[j*4 + k, :] = 0
                
    # Combine
    U_total = np.hstack([U44, U44])
    V_total = np.hstack([V44, V44_3])
    W_total = np.hstack([W44, W44])
    
    # Convert to lists
    U = U_total.tolist()
    V = V_total.tolist()
    W = W_total.tolist()
    
    return {"U": U, "V": V, "W": W}
