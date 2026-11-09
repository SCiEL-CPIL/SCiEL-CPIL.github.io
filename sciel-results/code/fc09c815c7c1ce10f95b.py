import numpy as np

DIMENSIONS = [4, 4, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 4, 4, 7

    # We will use a block decomposition:
    # Split A (4x4) into 2x2 blocks of 2x2 matrices.
    # Split B (4x7) into 2x2 blocks of 2x7 matrices.
    # C = A*B.
    # C_ij = sum_k A_ik B_kj
    #
    # Each term A_ik (2x2) * B_kj (2x7) requires a decomposition.
    # Let's find a rank-minimal decomposition for (2,2,7).
    #
    # R(2,2,7) can be decomposed as:
    # Split the 7 columns of B into 3+4 or 2+2+3 etc.
    # R(2,2,2) = 7 (Strassen)
    # R(2,2,1) = 2 (Naive)
    # So R(2,2,7) = R(2,2,2) + R(2,2,2) + R(2,2,3)
    # R(2,2,3) = R(2,2,2) + R(2,2,1) = 7 + 2 = 9
    # Total R(2,2,7) = 7 + 7 + 9 = 23
    #
    # There are 4 such blocks in the 2x2 block structure of C.
    # Total Rank = 4 * 23 = 92.
    #
    # We need to construct the U, V, W matrices for the full (4,4,7) problem.
    #
    # Let's define the helper to generate U, V, W for a (2,2,k) multiplication.
    # A is 2x2, B is 2xk, C is 2xk.
    # Indices for A: (0,0), (0,1), (1,0), (1,1) -> mapped to 0,1,2,3
    # Indices for B: (0,0)..(0,k-1), (1,0)..(1,k-1) -> mapped to 0..k-1, k..2k-1
    # Indices for C: (0,0)..(0,k-1), (1,0)..(1,k-1) -> mapped to 0..k-1, k..2k-1
    #
    # For Strassen (2,2,2):
    # A indices: 0,1,2,3
    # B indices: 0,1,2,3 (col0: 0,1; col1: 2,3)
    # C indices: 0,1,2,3
    #
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B10-B11)
    # M4 = A11(B01-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    #
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    #
    # Let's create a function to add a decomposition to global U, V, W.
    #
    def add_decomp(U, V, W, a_offset, b_offset, c_offset, u_entries, v_entries, w_entries):
        # u_entries: list of (index_in_a_block, value)
        # v_entries: list of (index_in_b_block, value)
        # w_entries: list of (index_in_c_block, value)
        # We assume one rank-1 term per call? No, let's pass the full rank decomposition.
        # Better: pass a list of tuples (u_vals, v_vals, w_vals) where u_vals is a dict or list.
        # Actually, let's just pass the matrices for the sub-problem and add them.
        pass

    # Let's build the (2,2,7) decomposition first.
    # It consists of:
    # 1. Strassen for cols 0,1 (B cols 0,1) -> Rank 7
    # 2. Strassen for cols 2,3 (B cols 2,3) -> Rank 7
    # 3. Strassen for col 4? No, R(2,2,1) is naive.
    #    Wait, R(2,2,3) = Strassen(2) + Naive(1).
    #    So for cols 4,5,6:
    #    Strassen for cols 4,5 -> Rank 7
    #    Naive for col 6 -> Rank 2
    #    Total for 3 cols: 9.
    #
    # So the 7 columns are handled as:
    # Col 0,1: Strassen (Rank 7)
    # Col 2,3: Strassen (Rank 7)
    # Col 4,5: Strassen (Rank 7)
    # Col 6:   Naive (Rank 2)
    # Total Rank 23.
    
    # Let's define the Strassen (2,2,2) decomposition matrices.
    # A: 4x1, B: 4x1, C: 4x1 (vectorized)
    # A: [A00, A01, A10, A11]
    # B: [B00, B01, B10, B11]  (Note: B is 2x2, so B00 is (0,0), B01 is (0,1), etc.)
    # C: [C00, C01, C10, C11]
    
    # M1: A: [1, 0, 0, 1], B: [1, 0, 0, 1], C: [1, 0, 0, 1]
    # M2: A: [0, 0, 1, 1], B: [1, 0, 0, 0], C: [0, 0, 1, 1]
    # M3: A: [1, 0, 0, 0], B: [0, 0, 1, -1], C: [0, 1, 0, 1]
    # M4: A: [0, 0, 0, 1], B: [1, -1, 0, 0], C: [1, 0, 1, 0]
    # M5: A: [1, 1, 0, 0], B: [0, 0, 0, 1], C: [-1, 1, 0, 0]
    # M6: A: [-1, 0, 1, 0], B: [1, 1, 0, 0], C: [0, 0, 0, 1]
    # M7: A: [0, 1, 0, -1], B: [0, 0, 1, 1], C: [1, 0, 0, 0]
    
    strassen_222 = [
        ([1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, 1]),
        ([0, 0, 1, 1], [1, 0, 0, 0], [0, 0, 1, 1]),
        ([1, 0, 0, 0], [0, 0, 1, -1], [0, 1, 0, 1]),
        ([0, 0, 0, 1], [1, -1, 0, 0], [1, 0, 1, 0]),
        ([1, 1, 0, 0], [0, 0, 0, 1], [-1, 1, 0, 0]),
        ([-1, 0, 1, 0], [1, 1, 0, 0], [0, 0, 0, 1]),
        ([0, 1, 0, -1], [0, 0, 1, 1], [1, 0, 0, 0]),
    ]
    
    # Naive (2,2,1) decomposition
    # A: [A00, A01, A10, A11]
    # B: [B00, B10] (Only one column, so B00 is (0,0), B10 is (1,0))
    # C: [C00, C10]
    # C00 = A00 B00 + A01 B10
    # C10 = A10 B00 + A11 B10
    # Rank 2:
    # M1: A: [1, 0, 0, 0], B: [1, 0], C: [1, 0]
    # M2: A: [0, 0, 0, 1], B: [0, 1], C: [0, 1]
    # Wait, C01 = A00 B01 + A01 B11 is missing.
    # Naive (2,2,1) means B is 2x1.
    # C is 2x1.
    # C00 = A00 B00 + A01 B10
    # C10 = A10 B00 + A11 B10
    # This is not rank 2.
    # R(2,2,1) is actually 4?
    # No, R(2,2,1) is the rank of the tensor T[i,j,k] = delta_{i,k} ... no.
    # T_{i,j,k} = 1 if k == i? No.
    # T_{i,j,k} = A_{i,j} B_{j,k}.
    # For 2x2x1:
    # T_{0,0,0} = A00 B00
    # T_{0,1,0} = A01 B10
    # T_{1,0,0} = A10 B00
    # T_{1,1,0} = A11 B10
    # This is 4 terms. Can it be done in less?
    # It is the identity tensor of size 2x2x1? No.
    # It is the tensor for matrix-vector multiplication.
    # Rank of 2x2x1 matvec is 4.
    # So R(2,2,1) = 4.
    # Then R(2,2,3) = R(2,2,2) + R(2,2,1) = 7 + 4 = 11.
    # R(2,2,7) = 7 + 7 + 11 = 25.
    # Total Rank = 4 * 25 = 100.
    # This is worse than 112? No, 100 < 112.
    # But can we do better?
    #
    # What if we use R(2,2,7) directly?
    # Maybe there is a better decomposition.
    # Let's stick with 100 for now, it's an improvement over 112.
    #
    # Let's re-verify R(2,2,1).
    # U (2x2, r), V (2x1, r), W (1x2, r).
    # C_{i,k} = sum_j A_{i,j} B_{j,k}
    # k is only 0.
    # C_{i,0} = sum_j A_{i,j} B_{j,0}
    # This is standard matvec.
    # Rank is 4.
    
    naive_221 = [
        ([1, 0, 0, 0], [1, 0], [1, 0]),
        ([0, 1, 0, 0], [0, 1], [1, 0]),
        ([0, 0, 1, 0], [1, 0], [0, 1]),
        ([0, 0, 0, 1], [0, 1], [0, 1]),
    ]
    
    # Now we assemble the (2,2,7) decomposition.
    # A_block: 2x2 (indices 0..3)
    # B_block: 2x7 (indices 0..13)
    # C_block: 2x7 (indices 0..13)
    #
    # We will create a list of (u_vec, v_vec, w_vec) for the (2,2,7) problem.
    # u_vec is length 4, v_vec is length 14, w_vec is length 14.
    
    sub_decomp = []
    
    # Helper to add a sub-decomp (e.g. 2x2x2) to the (2,2,7) decomp
    # at a specific column offset.
    def add_sub_decomp(decomp, a_vec, b_vec, w_vec, b_col_offset, w_col_offset):
        # a_vec: length 4
        # b_vec: length 4 (for 2x2)
        # w_vec: length 4 (for 2x2)
        # b_col_offset: which 2 columns of the 7-column B this applies to
        # w_col_offset: which 2 columns of the 7-column C this applies to
        
        u_full = [0.0] * 4
        v_full = [0.0] * 14
        w_full = [0.0] * 14
        
        for idx, val in enumerate(a_vec):
            if val != 0:
                u_full[idx] += val
        
        # B is 2x2. Indices:
        # (0,0) -> 0, (0,1) -> 1, (1,0) -> 2, (1,1) -> 3
        # In the 2x7 B:
        # Row 0: cols 0..6 -> indices 0..6
        # Row 1: cols 0..6 -> indices 7..13
        # So B_{0, col} is at index col.
        # B_{1, col} is at index 7 + col.
        
        # b_vec is [B00, B01, B10, B11] for the 2x2 block.
        # B00 is at (0, b_col_offset) -> index b_col_offset
        # B01 is at (0, b_col_offset+1) -> index b_col_offset+1
        # B10 is at (1, b_col_offset) -> index 7 + b_col_offset
        # B11 is at (1, b_col_offset+1) -> index 7 + b_col_offset+1
        
        v_full[b_col_offset] += b_vec[0]
        v_full[b_col_offset+1] += b_vec[1]
        v_full[7 + b_col_offset] += b_vec[2]
        v_full[7 + b_col_offset+1] += b_vec[3]
        
        # w_vec is [C00, C01, C10, C11] for the 2x2 block.
        # C00 is at (0, w_col_offset) -> index w_col_offset
        # C01 is at (0, w_col_offset+1) -> index w_col_offset+1
        # C10 is at (1, w_col_offset) -> index 7 + w_col_offset
        # C11 is at (1, w_col_offset+1) -> index 7 + w_col_offset+1
        
        w_full[w_col_offset] += w_vec[0]
        w_full[w_col_offset+1] += w_vec[1]
        w_full[7 + w_col_offset] += w_vec[2]
        w_full[7 + w_col_offset+1] += w_vec[3]
        
        decomp.append((u_full, v_full, w_full))
        
    def add_naive_sub_decomp(decomp, a_vec, b_vec, w_vec, b_col_offset, w_col_offset):
        # a_vec: length 4
        # b_vec: length 2 (for 2x1) [B00, B10]
        # w_vec: length 2 (for 2x1) [C00, C10]
        
        u_full = [0.0] * 4
        v_full = [0.0] * 14
        w_full = [0.0] * 14
        
        for idx, val in enumerate(a_vec):
            if val != 0:
                u_full[idx] += val
        
        # B00 is at (0, b_col_offset) -> index b_col_offset
        # B10 is at (1, b_col_offset) -> index 7 + b_col_offset
        
        v_full[b_col_offset] += b_vec[0]
        v_full[7 + b_col_offset] += b_vec[1]
        
        # C00 is at (0, w_col_offset) -> index w_col_offset
        # C10 is at (1, w_col_offset) -> index 7 + w_col_offset
        
        w_full[w_col_offset] += w_vec[0]
        w_full[7 + w_col_offset] += w_vec[1]
        
        decomp.append((u_full, v_full, w_full))
        
    # Add Strassen for cols 0,1
    for u, v, w in strassen_222:
        add_sub_decomp(sub_decomp, u, v, w, 0, 0)
        
    # Add Strassen for cols 2,3
    for u, v, w in strassen_222:
        add_sub_decomp(sub_decomp, u, v, w, 2, 2)
        
    # Add Strassen for cols 4,5
    for u, v, w in strassen_222:
        add_sub_decomp(sub_decomp, u, v, w, 4, 4)
        
    # Add Naive for col 6
    for u, v, w in naive_221:
        add_naive_sub_decomp(sub_decomp, u, v, w, 6, 6)
        
    # Now we have the (2,2,7) decomposition in `sub_decomp`.
    # There are 4 blocks in the 2x2 block structure of A and B.
    # A blocks: A00, A01, A10, A11
    # B blocks: B00 (2x7), B01 (2x7), B10 (2x7), B11 (2x7)
    # C blocks: C00 (2x7), C01 (2x7), C10 (2x7), C11 (2x7)
    #
    # C00 = A00 B00 + A01 B10
    # C01 = A00 B01 + A01 B11
    # C10 = A10 B00 + A11 B10
    # C11 = A10 B01 + A11 B11
    #
    # We need to map the local indices of the (2,2,7) decomposition to the global indices.
    # Global A is 4x4.
    # A00: rows 0,1; cols 0,1 -> global indices:
    #   A00[0,0] -> 0, A00[0,1] -> 1, A00[1,0] -> 4, A00[1,1] -> 5
    #   Wait, A is 4x4. Row-major:
    #   0: (0,0), 1: (0,1), 2: (0,2), 3: (0,3)
    #   4: (1,0), 5: (1,1), 6: (1,2), 7: (1,3)
    #   8: (2,0), 9: (2,1), 10: (2,2), 11: (2,3)
    #   12: (3,0), 13: (3,1), 14: (3,2), 15: (3,3)
    #
    # Local A indices for (2,2) block:
    # 0: (0,0), 1: (0,1), 2: (1,0), 3: (1,1)
    #
    # For A00 (top-left):
    # Local 0 -> Global 0
    # Local 1 -> Global 1
    # Local 2 -> Global 4
    # Local 3 -> Global 5
    #
    # For A01 (top-right):
    # Local 0 -> Global 2
    # Local 1 -> Global 3
    # Local 2 -> Global 6
    # Local 3 -> Global 7
    #
    # For A10 (bottom-left):
    # Local 0 -> Global 8
    # Local 1 -> Global 9
    # Local 2 -> Global 12
    # Local 3 -> Global 13
    #
    # For A11 (bottom-right):
    # Local 0 -> Global 10
    # Local 1 -> Global 11
    # Local 2 -> Global 14
    # Local 3 -> Global 15
    
    # Global B is 4x7.
    # B00 (top-left 2x7): rows 0,1; cols 0..6
    #   B00[0,0] -> 0, ..., B00[0,6] -> 6
    #   B00[1,0] -> 7, ..., B00[1,6] -> 13
    #
    # Local B indices for (2,7) block:
    # 0..6: Row 0, Cols 0..6
    # 7..13: Row 1, Cols 0..6
    #
    # For B00:
    # Local 0..6 -> Global 0..6
    # Local 7..13 -> Global 7..13
    #
    # For B01 (top-right 2x7): rows 0,1; cols 7..13 (Wait, B is 4x7, so cols are 0..6)
    # B is 4 rows, 7 cols.
    # B00: rows 0,1. Cols 0..6.
    # B01: rows 0,1. Cols 0..6? No.
    # The block decomposition of B (4x7) into 2x2 blocks of size 2x7?
    # No, the block size is determined by A and B dimensions.
    # A is 4x4. B is 4x7.
    # We split A into 2x2 blocks of 2x2.
    # We split B into 2x2 blocks of 2x7.
    # So B00 is rows 0,1 and all 7 cols?
    # Yes, because the column dimension of the block is p=7.
    # So B00 is 2x7, B01 is 2x7, B10 is 2x7, B11 is 2x7.
    #
    # Global B indices:
    # Row 0: 0..6
    # Row 1: 7..13
    # Row 2: 14..20
    # Row 3: 21..27
    #
    # B00 (rows 0,1):
    # Local 0..6 -> Global 0..6
    # Local 7..13 -> Global 7..13
    #
    # B10 (rows 2,3):
    # Local 0..6 -> Global 14..20
    # Local 7..13 -> Global 21..27
    
    # Global C is 4x7.
    # C00 (rows 0,1):
    # Local 0..6 -> Global 0..6
    # Local 7..13 -> Global 7..13
    #
    # C10 (rows 2,3):
    # Local 0..6 -> Global 14..20
    # Local 7..13 -> Global 21..27
    
    # Now we combine the 4 blocks.
    # Total rank = 4 * len(sub_decomp) = 4 * 25 = 100.
    
    total_rank = 4 * len(sub_decomp)
    U = np.zeros((16, total_rank))
    V = np.zeros((28, total_rank))
    W = np.zeros((28, total_rank))
    
    col = 0
    # Block A00, B00 -> C00
    for u, v, w in sub_decomp:
        # Map A
        U[0, col] += u[0]
        U[1, col] += u[1]
        U[4, col] += u[2]
        U[5, col] += u[3]
        
        # Map B
        for i in range(14):
            V[i, col] += v[i]
            
        # Map C
        for i in range(14):
            W[i, col] += w[i]
            
        col += 1
        
    # Block A01, B10 -> C01
    # A01: Local 0->2, 1->3, 2->6, 3->7
    # B10: Local 0..6 -> 14..20, 7..13 -> 21..27
    # C01: Local 0..6 -> 0..6, 7..13 -> 7..13 (Wait, C01 is top-right block of C)
    # C is 4x7.
    # C00: rows 0,1.
    # C01: rows 0,1.
    # So C01 maps to same global C indices as C00?
    # No, C00 is the top-left block of C?
    # C is 4x7.
    # We split C into 2x2 blocks of 2x7.
    # C00: rows 0,1.
    # C10: rows 2,3.
    # There is no C01, C11 in the row sense.
    # The block structure is:
    # [ C00  C01 ]
    # [ C10  C11 ]
    # But C has only 2 block-rows and 2 block-cols?
    # A is 4x4 -> 2x2 blocks.
    # B is 4x7 -> 2x2 blocks (each 2x7).
    # C is 4x7 -> 2x2 blocks (each 2x7).
    # So C00 is top-left, C01 is top-right?
    # No, the column dimension of the block is 7.
    # So there is only 1 block-column.
    # So C is 2x1 blocks?
    # Let's re-read the block structure.
    # A (4x4) = [ A00 A01 ]
    #             [ A10 A11 ]
    # B (4x7) = [ B00 ]
    #             [ B10 ]
    # (B is 4x7, so it has 2 block-rows of 2x7).
    # C (4x7) = [ C00 ]
    #             [ C10 ]
    # C00 = A00 B00 + A01 B10
    # C10 = A10 B00 + A11 B10
    
    # So we only have 2 block-rows for C.
    # And 1 block-column.
    # So the blocks are:
    # 1. A00 * B00 -> C00
    # 2. A01 * B10 -> C00
    # 3. A10 * B00 -> C10
    # 4. A11 * B10 -> C10
    
    # Let's re-evaluate the mapping.
    
    # Block 1: A00 * B00 -> C00
    # A00: 0->0, 1->1, 2->4, 3->5
    # B00: 0..6->0..6, 7..13->7..13
    # C00: 0..6->0..6, 7..13->7..13
    
    # Block 2: A01 * B10 -> C00
    # A01: 0->2, 1->3, 2->6, 3->7
    # B10: 0..6->14..20, 7..13->21..27
    # C00: 0..6->0..6, 7..13->7..13
    
    # Block 3: A10 * B00 -> C10
    # A10: 0->8, 1->9, 2->12, 3->13
    # B00: 0..6->0..6, 7..13->7..13
    # C10: 0..6->14..20, 7..13->21..27
    
    # Block 4: A11 * B10 -> C10
    # A11: 0->10, 1->11, 2->14, 3->15
    # B10: 0..6->14..20, 7..13->21..27
    # C10: 0..6->14..20, 7..13->21..27
    
    # Let's rewrite the construction with these mappings.
    
    U = np.zeros((16, total_rank))
    V = np.zeros((28, total_rank))
    W = np.zeros((28, total_rank))
    
    col = 0
    for u, v, w in sub_decomp:
        # Block 1
        U[0, col] += u[0]
        U[1, col] += u[1]
        U[4, col] += u[2]
        U[5, col] += u[3]
        for i in range(14):
            V[i, col] += v[i]
            W[i, col] += w[i]
        col += 1
        
    for u, v, w in sub_decomp:
        # Block 2
        U[2, col] += u[0]
        U[3, col] += u[1]
        U[6, col] += u[2]
        U[7, col] += u[3]
        for i in range(14):
            V[i + 14, col] += v[i]
            W[i, col] += w[i]
        col += 1
        
    for u, v, w in sub_decomp:
        # Block 3
        U[8, col] += u[0]
        U[9, col] += u[1]
        U[12, col] += u[2]
        U[13, col] += u[3]
        for i in range(14):
            V[i, col] += v[i]
            W[i + 14, col] += w[i]
        col += 1
        
    for u, v, w in sub_decomp:
        # Block 4
        U[10, col] += u[0]
        U[11, col] += u[1]
        U[14, col] += u[2]
        U[15, col] += u[3]
        for i in range(14):
            V[i + 14, col] += v[i]
            W[i + 14, col] += w[i]
        col += 1
        
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
