import numpy as np

DIMENSIONS = [4, 4, 7]

def construct():
    n, m, p = DIMENSIONS  # 4, 4, 7
    
    # Total rank
    rank_444 = 49
    rank_443 = 44
    total_rank = rank_444 + rank_443
    
    U = np.zeros((n*m, total_rank))
    V = np.zeros((m*p, total_rank))
    W = np.zeros((p*n, total_rank))
    
    # --- Part 1: 4x4 * 4x4 (Strassen) ---
    # Columns 0..3 of B
    # We build the 4x4x4 Strassen decomposition and map it to the full indices.
    
    # Strassen for 4x4x4:
    # A is 4x4, B is 4x4, C is 4x4.
    # Split into 2x2 blocks of 2x2 matrices.
    # A11, A12, A21, A22
    # B11, B12, B21, B22
    # C11, C12, C21, C22
    
    # Indices for U (A): i*m + j. m=4.
    # A11: i in 0,1; j in 0,1 -> indices 0,1,4,5
    # A12: i in 0,1; j in 2,3 -> indices 2,3,6,7
    # A21: i in 2,3; j in 0,1 -> indices 8,9,12,13
    # A22: i in 2,3; j in 2,3 -> indices 10,11,14,15
    
    # Indices for V (B): j*p + k. p=7.
    # For 4x4 part, k in 0..3.
    # B11: j in 0,1; k in 0,1 -> indices 0*7+0, 0*7+1, 1*7+0, 1*7+1 -> 0,1,7,8
    # B12: j in 0,1; k in 2,3 -> indices 0*7+2, 0*7+3, 1*7+2, 1*7+3 -> 2,3,9,10
    # B21: j in 2,3; k in 0,1 -> indices 2*7+0, 2*7+1, 3*7+0, 3*7+1 -> 14,15,21,22
    # B22: j in 2,3; k in 2,3 -> indices 2*7+2, 2*7+3, 3*7+2, 3*7+3 -> 16,17,23,24
    
    # Indices for W (C): k*n + i. n=4.
    # C11: k in 0,1; i in 0,1 -> indices 0*4+0, 0*4+1, 1*4+0, 1*4+1 -> 0,1,4,5
    # C12: k in 0,1; i in 2,3 -> indices 0*4+2, 0*4+3, 1*4+2, 1*4+3 -> 2,3,6,7
    # C21: k in 2,3; i in 0,1 -> indices 2*4+0, 2*4+1, 3*4+0, 3*4+1 -> 8,9,12,13
    # C22: k in 2,3; i in 2,3 -> indices 2*4+2, 2*4+3, 3*4+2, 3*4+3 -> 10,11,14,15
    
    A11 = [0, 1, 4, 5]
    A12 = [2, 3, 6, 7]
    A21 = [8, 9, 12, 13]
    A22 = [10, 11, 14, 15]
    
    B11 = [0, 1, 7, 8]
    B12 = [2, 3, 9, 10]
    B21 = [14, 15, 21, 22]
    B22 = [16, 17, 23, 24]
    
    C11 = [0, 1, 4, 5]
    C12 = [2, 3, 6, 7]
    C21 = [8, 9, 12, 13]
    C22 = [10, 11, 14, 15]
    
    # Strassen patterns for 2x2*2x2 (rank 7)
    # Each pattern is (U_indices, V_indices, W_indices) with signs
    # Local indices 0..3 for 2x2 blocks.
    # We need to map local 2x2 indices to global 4x4 block indices.
    
    # Helper to add a 2x2x2 Strassen pattern to a specific block set
    def add_strassen_2x2(u_blocks, v_blocks, w_blocks, base_col):
        # u_blocks: list of 4 lists of global U indices (A11, A12, A21, A22)
        # v_blocks: list of 4 lists of global V indices (B11, B12, B21, B22)
        # w_blocks: list of 4 lists of global W indices (C11, C12, C21, C22)
        
        # Patterns:
        # 1: (A11+A22)(B11+B22) -> C11+=, C22+=
        # 2: (A21+A22)B11 -> C21+=
        # 3: A11(B12-B22) -> C12+=
        # 4: A22(B21-B11) -> C11+=, C21+=
        # 5: (A11+A12)B22 -> C11-=, C12+=
        # 6: (A21-A11)(B11+B12) -> C22+=
        # 7: (A12-A22)(B21+B22) -> C11+=
        
        col = base_col
        
        # M1
        for idx in u_blocks[0]: U[idx, col] += 1
        for idx in u_blocks[3]: U[idx, col] += 1
        for idx in v_blocks[0]: V[idx, col] += 1
        for idx in v_blocks[3]: V[idx, col] += 1
        for idx in w_blocks[0]: W[idx, col] += 1
        for idx in w_blocks[3]: W[idx, col] += 1
        col += 1
        
        # M2
        for idx in u_blocks[2]: U[idx, col] += 1
        for idx in u_blocks[3]: U[idx, col] += 1
        for idx in v_blocks[0]: V[idx, col] += 1
        for idx in w_blocks[2]: W[idx, col] += 1
        col += 1
        
        # M3
        for idx in u_blocks[0]: U[idx, col] += 1
        for idx in v_blocks[1]: V[idx, col] += 1
        for idx in v_blocks[3]: V[idx, col] -= 1
        for idx in w_blocks[1]: W[idx, col] += 1
        col += 1
        
        # M4
        for idx in u_blocks[3]: U[idx, col] += 1
        for idx in v_blocks[2]: V[idx, col] += 1
        for idx in v_blocks[0]: V[idx, col] -= 1
        for idx in w_blocks[0]: W[idx, col] += 1
        for idx in w_blocks[2]: W[idx, col] += 1
        col += 1
        
        # M5
        for idx in u_blocks[0]: U[idx, col] += 1
        for idx in u_blocks[1]: U[idx, col] += 1
        for idx in v_blocks[3]: V[idx, col] += 1
        for idx in w_blocks[0]: W[idx, col] -= 1
        for idx in w_blocks[1]: W[idx, col] += 1
        col += 1
        
        # M6
        for idx in u_blocks[2]: U[idx, col] += 1
        for idx in u_blocks[0]: U[idx, col] -= 1
        for idx in v_blocks[0]: V[idx, col] += 1
        for idx in v_blocks[1]: V[idx, col] += 1
        for idx in w_blocks[3]: W[idx, col] += 1
        col += 1
        
        # M7
        for idx in u_blocks[1]: U[idx, col] += 1
        for idx in u_blocks[3]: U[idx, col] -= 1
        for idx in v_blocks[2]: V[idx, col] += 1
        for idx in v_blocks[3]: V[idx, col] += 1
        for idx in w_blocks[0]: W[idx, col] += 1
        col += 1
        
        return col

    # Apply Strassen for 4x4x4
    # The 4x4 blocks are composed of 2x2 sub-blocks.
    # We treat the 4x4 product as 7 products of 2x2 matrices.
    # Each 2x2 product is itself a Strassen product of 2x1? No.
    # The 4x4 Strassen uses 7 macro-terms.
    # Each macro-term is a product of two 2x2 matrices.
    # We can decompose each 2x2x2 product with rank 7.
    # So total rank 7*7=49.
    
    # We need to define the 2x2 sub-blocks for the macro-terms.
    # A11 (2x2) consists of 4 1x1? No, we are doing 4x4 -> 2x2 blocks.
    # The "2x2" in Strassen refers to blocks of size 2x2.
    # So A11 is a 2x2 matrix.
    # To decompose A11 * B11, we use Strassen on the 2x2 matrices?
    # No, Strassen on 2x2 matrices gives rank 7.
    # So we just apply the rank 7 decomposition to the 2x2 blocks.
    
    # The indices A11, etc., are lists of 4 indices (for the 4 elements of the 2x2 block).
    # We need to apply the 7 Strassen patterns to these 4-element lists.
    
    # Wait, the 4x4 Strassen algorithm computes 7 terms like (A11+A22)(B11+B22).
    # Here A11, A22 are 2x2 matrices.
    # The product (A11+A22)(B11+B22) is a 2x2 matrix product.
    # We can decompose this 2x2 matrix product using Strassen (rank 7).
    # So for each of the 7 macro-terms, we add 7 rank-1 terms.
    # Total 49 terms.
    
    # We need to map the 2x2 block indices to the 1x1 element indices.
    # A11 elements are at U[0], U[1], U[4], U[5].
    # Local 2x2 indices: 0->(0,0), 1->(0,1), 2->(1,0), 3->(1,1).
    # So A11_local[0] = U[0], A11_local[1] = U[1], A11_local[2] = U[4], A11_local[3] = U[5].
    
    # Let's define the local index mappings for the 2x2 blocks.
    # For A:
    # Block 0 (A11): [0, 1, 4, 5]
    # Block 1 (A12): [2, 3, 6, 7]
    # Block 2 (A21): [8, 9, 12, 13]
    # Block 3 (A22): [10, 11, 14, 15]
    
    # For B (cols 0-3):
    # Block 0 (B11): [0, 1, 7, 8]
    # Block 1 (B12): [2, 3, 9, 10]
    # Block 2 (B21): [14, 15, 21, 22]
    # Block 3 (B22): [16, 17, 23, 24]
    
    # For C (cols 0-3):
    # Block 0 (C11): [0, 1, 4, 5]
    # Block 1 (C12): [2, 3, 6, 7]
    # Block 2 (C21): [8, 9, 12, 13]
    # Block 3 (C22): [10, 11, 14, 15]
    
    # We will iterate through the 7 Strassen macro-terms.
    # For each macro-term, we identify the A and B blocks involved and their signs.
    # Then we apply the 7 Strassen sub-terms for 2x2x2 to those blocks.
    
    # Macro-term definitions: (A_blocks, A_signs, B_blocks, B_signs, C_blocks, C_signs)
    # A_blocks: indices into the list of 4 A-blocks.
    
    macros_444 = [
        # M1: (A11+A22)(B11+B22) -> C11+, C22+
        ([0, 3], [1, 1], [0, 3], [1, 1], [(0, 1), (3, 1)]),
        # M2: (A21+A22)B11 -> C21+
        ([2, 3], [1, 1], [0], [1], [(2, 1)]),
        # M3: A11(B12-B22) -> C12+
        ([0], [1], [1, 3], [1, -1], [(1, 1)]),
        # M4: A22(B21-B11) -> C11+, C21+
        ([3], [1], [2, 0], [1, -1], [(0, 1), (2, 1)]),
        # M5: (A11+A12)B22 -> C11-, C12+
        ([0, 1], [1, 1], [3], [1], [(0, -1), (1, 1)]),
        # M6: (A21-A11)(B11+B12) -> C22+
        ([2, 0], [1, -1], [0, 1], [1, 1], [(3, 1)]),
        # M7: (A12-A22)(B21+B22) -> C11+
        ([1, 3], [1, -1], [2, 3], [1, 1], [(0, 1)]),
    ]
    
    # Strassen sub-patterns for 2x2x2 (rank 7)
    # Local indices 0..3 for 2x2 blocks.
    sub_patterns = [
        # (U: [(idx, sign)], V: [(idx, sign)], W: [(idx, sign)])
        ([(0, 1), (3, 1)], [(0, 1), (3, 1)], [(0, 1), (3, 1)]),  # M1
        ([(2, 1), (3, 1)], [(0, 1)], [(2, 1)]),                    # M2
        ([(0, 1)], [(1, 1), (3, -1)], [(1, 1)]),                   # M3
        ([(3, 1)], [(2, 1), (0, -1)], [(0, 1), (2, 1)]),           # M4
        ([(0, 1), (1, 1)], [(3, 1)], [(0, -1), (1, 1)]),           # M5
        ([(2, 1), (0, -1)], [(0, 1), (1, 1)], [(3, 1)]),           # M6
        ([(1, 1), (3, -1)], [(2, 1), (3, 1)], [(0, 1)]),           # M7
    ]
    
    col = 0
    A_blocks_global = [A11, A12, A21, A22]
    B_blocks_global = [B11, B12, B21, B22]
    C_blocks_global = [C11, C12, C21, C22]
    
    for macro in macros_444:
        a_idxs, a_signs, b_idxs, b_signs, c_targets = macro
        for sub in sub_patterns:
            u_entries, v_entries, w_entries = sub
            
            # U entries
            for local_idx, sign in u_entries:
                # Which macro-block is this?
                # The sub-pattern local indices 0..3 correspond to the 2x2 blocks of the macro-term.
                # Wait, the macro-term combines multiple 2x2 blocks.
                # Example M1: A11+A22.
                # The "2x2 matrix" being multiplied is (A11+A22).
                # Its elements are sums of elements of A11 and A22.
                # Local index 0 of (A11+A22) corresponds to A11[0] + A22[0].
                # So for each local index in the sub-pattern, we need to add contributions from all macro-blocks involved.
                
                for mi, a_sign in zip(a_idxs, a_signs):
                    global_block = A_blocks_global[mi]
                    global_idx = global_block[local_idx]
                    U[global_idx, col] += a_sign * sign
            
            # V entries
            for local_idx, sign in v_entries:
                for mi, b_sign in zip(b_idxs, b_signs):
                    global_block = B_blocks_global[mi]
                    global_idx = global_block[local_idx]
                    V[global_idx, col] += b_sign * sign
            
            # W entries
            for local_idx, sign in w_entries:
                # C targets: list of (block_idx, sign)
                for c_block_idx, c_sign in c_targets:
                    global_block = C_blocks_global[c_block_idx]
                    global_idx = global_block[local_idx]
                    W[global_idx, col] += c_sign * sign
            
            col += 1
    
    # --- Part 2: 4x4 * 4x3 (Optimized) ---
    # Columns 4..6 of B.
    # j in 0..3, k in 4..6.
    # Split j into 0,1 and 2,3.
    # Term 1: A[:,0:2] * B[0:2, 4:7]
    # Term 2: A[:,2:4] * B[2:4, 4:7]
    
    # Each term is 4x2 * 2x3.
    # Split A_part (4x2) into A_top (2x2) and A_bot (2x2).
    # A_top * B_part (2x3) -> 2x2 * 2x3 (Rank 11)
    # A_bot * B_part (2x3) -> 2x2 * 2x3 (Rank 11)
    # Total 22 per j-split.
    # Two j-splits -> 44.
    
    # Indices for A:
    # A_top (j=0,1):
    #   i=0,1; j=0,1 -> 0,1,4,5
    #   i=2,3; j=0,1 -> 8,9,12,13
    # A_bot (j=2,3):
    #   i=0,1; j=2,3 -> 2,3,6,7
    #   i=2,3; j=2,3 -> 10,11,14,15
    
    # Indices for B (k=4,5,6):
    # B_top (j=0,1):
    #   j=0; k=4,5,6 -> 0*7+4, 0*7+5, 0*7+6 -> 4,5,6
    #   j=1; k=4,5,6 -> 1*7+4, 1*7+5, 1*7+6 -> 11,12,13
    # B_bot (j=2,3):
    #   j=2; k=4,5,6 -> 2*7+4, 2*7+5, 2*7+6 -> 18,19,20
    #   j=3; k=4,5,6 -> 3*7+4, 3*7+5, 3*7+6 -> 25,26,27
    
    # Indices for C (k=4,5,6):
    # C_top (i=0,1):
    #   k=4; i=0,1 -> 4*4+0, 4*4+1 -> 16,17
    #   k=5; i=0,1 -> 5*4+0, 5*4+1 -> 20,21
    #   k=6; i=0,1 -> 6*4+0, 6*4+1 -> 24,25
    # C_bot (i=2,3):
    #   k=4; i=2,3 -> 4*4+2, 4*4+3 -> 18,19
    #   k=5; i=2,3 -> 5*4+2, 5*4+3 -> 22,23
    #   k=6; i=2,3 -> 6*4+2, 6*4+3 -> 26,27
    
    A_top_top = [0, 1, 4, 5]
    A_top_bot = [8, 9, 12, 13]
    A_bot_top = [2, 3, 6, 7]
    A_bot_bot = [10, 11, 14, 15]
    
    B_top = [4, 5, 6, 11, 12, 13] # j=0,1; k=4,5,6. Order: j=0(k4,5,6), j=1(k4,5,6)
    B_bot = [18, 19, 20, 25, 26, 27]
    
    C_top = [16, 17, 20, 21, 24, 25] # k=4,5,6; i=0,1. Order: k=4(i0,1), k=5(i0,1), k=6(i0,1)
    C_bot = [18, 19, 22, 23, 26, 27]
    
    # We need a rank 11 decomposition for 2x2 * 2x3.
    # A (2x2), B (2x3), C (2x3).
    # Split B into B1 (col 0) and B2 (cols 1,2).
    # C1 = A B1 (2x1). Rank 4.
    # C2 = A B2 (2x2). Rank 7.
    
    # Local indices for 2x2 A: 0,1,2,3.
    # Local indices for 2x3 B: 0,1,2 (j=0) and 3,4,5 (j=1).
    # Local indices for 2x3 C: 0,1 (k=0), 2,3 (k=1), 4,5 (k=2).
    
    # Rank 4 for 2x2 * 2x1:
    # A00 B0 -> C0
    # A01 B1 -> C0
    # A10 B0 -> C1
    # A11 B1 -> C1
    
    # Rank 7 for 2x2 * 2x2:
    # Standard Strassen.
    
    # We will add these to the main matrices.
    
    def add_2x2_2x3(u_top, u_bot, v_block, w_top, w_bot, base_col):
        # u_top, u_bot: lists of 4 indices for the 2x2 A blocks
        # v_block: list of 6 indices for the 2x3 B block
        # w_top, w_bot: lists of 6 indices for the 2x3 C blocks
        
        col = base_col
        
        # Part 1: 2x2 * 2x1 (Col 0 of B, Col 0 of C)
        # B col 0 indices: v_block[0], v_block[3]
        # C col 0 indices: w_top[0], w_bot[0]
        
        # A00 * B00 -> C00
        U[u_top[0], col] = 1; V[v_block[0], col] = 1; W[w_top[0], col] = 1
        col += 1
        # A01 * B10 -> C00
        U[u_top[1], col] = 1; V[v_block[3], col] = 1; W[w_top[0], col] = 1
        col += 1
        # A10 * B00 -> C10
        U[u_bot[0], col] = 1; V[v_block[0], col] = 1; W[w_bot[0], col] = 1
        col += 1
        # A11 * B10 -> C10
        U[u_bot[1], col] = 1; V[v_block[3], col] = 1; W[w_bot[0], col] = 1
        col += 1
        
        # Part 2: 2x2 * 2x2 (Cols 1,2 of B, Cols 1,2 of C)
        # B cols 1,2 indices: v_block[1], v_block[2] (j=0), v_block[4], v_block[5] (j=1)
        # C cols 1,2 indices: w_top[2], w_top[3] (k=1), w_top[4], w_top[5] (k=2)
        
        # Strassen for 2x2 * 2x2
        # Local B indices: 0->j=0,k=1; 1->j=0,k=2; 2->j=1,k=1; 3->j=1,k=2
        # Local C indices: 0->i=0,k=1; 1->i=1,k=1; 2->i=0,k=2; 3->i=1,k=2
        
        b_local_to_global = [v_block[1], v_block[2], v_block[4], v_block[5]]
        c_local_to_global = [w_top[2], w_bot[2], w_top[4], w_bot[4]] # Wait, C is 2x3.
        # C indices:
        # k=1 (col 1): w_top[2] (i=0), w_bot[2] (i=1)
        # k=2 (col 2): w_top[4] (i=0), w_bot[4] (i=1)
        # So C11 (local 0) = w_top[2]
        # C12 (local 1) = w_top[4]
        # C21 (local 2) = w_bot[2]
        # C22 (local 3) = w_bot[4]
        
        c_local_to_global = [w_top[2], w_top[4], w_bot[2], w_bot[4]]
        
        # Apply Strassen patterns
        # M1: (A11+A22)(B11+B22) -> C11+, C22+
        for idx in u_top: U[idx, col] += 1 # A11 is top row of A?
        # Wait, u_top is the 2x2 block for i=0,1.
        # A11 (local 0,0) is u_top[0]
        # A12 (local 0,1) is u_top[1]
        # A21 (local 1,0) is u_bot[0]
        # A22 (local 1,1) is u_bot[1]
        
        # Correct mapping:
        # A_local[0] = u_top[0]
        # A_local[1] = u_top[1]
        # A_local[2] = u_bot[0]
        # A_local[3] = u_bot[1]
        
        a_local_to_global = [u_top[0], u_top[1], u_bot[0], u_bot[1]]
        
        # Strassen patterns again
        # 1: (A00+A11)(B00+B11) -> C00+, C11+
        U[a_local_to_global[0], col] += 1
        U[a_local_to_global[3], col] += 1
        V[b_local_to_global[0], col] += 1
        V[b_local_to_global[3], col] += 1
        W[c_local_to_global[0], col] += 1
        W[c_local_to_global[3], col] += 1
        col += 1
        
        # 2: (A10+A11)B00 -> C10+
        U[a_local_to_global[2], col] += 1
        U[a_local_to_global[3], col] += 1
        V[b_local_to_global[0], col] += 1
        W[c_local_to_global[2], col] += 1
        col += 1
        
        # 3: A00(B01-B11) -> C01+
        U[a_local_to_global[0], col] += 1
        V[b_local_to_global[1], col] += 1
        V[b_local_to_global[3], col] -= 1
        W[c_local_to_global[1], col] += 1
        col += 1
        
        # 4: A11(B10-B00) -> C00+, C10+
        U[a_local_to_global[3], col] += 1
        V[b_local_to_global[2], col] += 1
        V[b_local_to_global[0], col] -= 1
        W[c_local_to_global[0], col] += 1
        W[c_local_to_global[2], col] += 1
        col += 1
        
        # 5: (A00+A01)B11 -> C00-, C01+
        U[a_local_to_global[0], col] += 1
        U[a_local_to_global[1], col] += 1
        V[b_local_to_global[3], col] += 1
        W[c_local_to_global[0], col] -= 1
        W[c_local_to_global[1], col] += 1
        col += 1
        
        # 6: (A10-A00)(B00+B01) -> C11+
        U[a_local_to_global[2], col] += 1
        U[a_local_to_global[0], col] -= 1
        V[b_local_to_global[0], col] += 1
        V[b_local_to_global[1], col] += 1
        W[c_local_to_global[3], col] += 1
        col += 1
        
        # 7: (A01-A11)(B10+B11) -> C00+
        U[a_local_to_global[1], col] += 1
        U[a_local_to_global[3], col] -= 1
        V[b_local_to_global[2], col] += 1
        V[b_local_to_global[3], col] += 1
        W[c_local_to_global[0], col] += 1
        col += 1
        
        return col
    
    # Apply to Term 1 (j=0,1)
    col = add_2x2_2x3(A_top_top, A_top_bot, B_top, C_top, C_bot, col)
    
    # Apply to Term 2 (j=2,3)
    col = add_2x2_2x3(A_bot_top, A_bot_bot, B_bot, C_top, C_bot, col)
    
    return {"U": U, "V": V, "W": W}

