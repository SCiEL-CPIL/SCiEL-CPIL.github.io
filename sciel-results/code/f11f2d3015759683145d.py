# EVOLVE-BLOCK-START
def construct():
    n, m, p = 4, 4, 7
    # U shape (n*m, rank) = (16, rank)
    # V shape (m*p, rank) = (28, rank)
    # W shape (p*n, rank) = (28, rank)
    
    rank_strassen = 49
    rank_schoolbook = 48
    total_rank = rank_strassen + rank_schoolbook
    
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]
    
    # Helper to get block indices
    # A is 4x4.
    # Block 11: (0,0), (0,1), (1,0), (1,1)
    # Block 12: (0,2), (0,3), (1,2), (1,3)
    # Block 21: (2,0), (2,1), (3,0), (3,1)
    # Block 22: (2,2), (2,3), (3,2), (3,3)
    
    def get_block_indices(block_id):
        # block_id: 0 for 11, 1 for 12, 2 for 21, 3 for 22
        if block_id == 0:
            return [(0,0), (0,1), (1,0), (1,1)]
        elif block_id == 1:
            return [(0,2), (0,3), (1,2), (1,3)]
        elif block_id == 2:
            return [(2,0), (2,1), (3,0), (3,1)]
        elif block_id == 3:
            return [(2,2), (2,3), (3,2), (3,3)]
        
    # Define the 7 outer Strassen combinations
    # Each is a list of (coeff, block_id, local_i, local_j)
    # A_s is a 2x2 matrix.
    # A_s[0,0] is the top-left block.
    # We need to express A_s[i,j] as a linear combination of A entries.
    
    # A_s[i,j] = sum_{blocks} c * A_block
    # For A, the blocks are 11, 12, 21, 22.
    # A_1 = A_11 + A_22
    # A_2 = A_21 + A_22
    # A_3 = A_11
    # A_4 = A_22
    # A_5 = A_11 + A_12
    # A_6 = A_21 - A_11
    # A_7 = A_12 - A_22
    
    # B_s similarly.
    
    # We will iterate s from 0 to 6.
    # For each s, we have A_s and B_s.
    # We compute the 7 Strassen terms for A_s * B_s.
    
    # Strassen terms for 2x2:
    # m1 = (A00+A11)(B00+B11)
    # m2 = (A10+A11)B00
    # m3 = A00(B01-B11)
    # m4 = A11(B10-B00)
    # m5 = (A00+A01)B11
    # m6 = (A10-A00)(B00+B01)
    # m7 = (A01-A11)(B10+B11)
    
    # We need to map the local 2x2 indices to the global 4x4 indices.
    # For A_s, the local index (li, lj) corresponds to a block.
    # For example, if A_s[0,0] = A_11, then local (0,0) maps to global block 11.
    # Within the block, the local 2x2 indices (0,0), (0,1), (1,0), (1,1) map to specific global indices.
    # Block 11: (0,0)->(0,0), (0,1)->(0,1), (1,0)->(1,0), (1,1)->(1,1)
    # Block 12: (0,0)->(0,2), (0,1)->(0,3), (1,0)->(1,2), (1,1)->(1,3)
    # etc.
    
    # Let's define the mapping for each block.
    # block_to_global = {
    #   0: {(0,0): (0,0), (0,1): (0,1), (1,0): (1,0), (1,1): (1,1)},
    #   1: {(0,0): (0,2), (0,1): (0,3), (1,0): (1,2), (1,1): (1,3)},
    #   2: {(0,0): (2,0), (0,1): (2,1), (1,0): (3,0), (1,1): (3,1)},
    #   3: {(0,0): (2,2), (0,1): (2,3), (1,0): (3,2), (1,1): (3,3)}
    # }
    
    # For each outer step s, we define A_s_blocks and B_s_blocks.
    # A_s_blocks is a 2x2 matrix where each entry is a list of (coeff, block_id).
    
    A_s_defs = [
        [[(1, 0), (1, 3)], [(0, 0), (0, 0)]], # A1: A11+A22, 0
        # Wait, A_s is a 2x2 matrix of blocks.
        # A_1 = [A_11+A_22, 0; 0, 0]? No.
        # The Strassen algorithm for 4x4 uses 7 multiplications of 2x2 matrices.
        # The inputs to these 2x2 multiplications are linear combinations of the 2x2 blocks of A and B.
        # M1 = (A11+A22)(B11+B22) -> A1 is a 2x2 matrix? No, A11+A22 is a 2x2 matrix.
        # So the "block" here is the 2x2 matrix itself.
        # So A_s is a 2x2 matrix, where each entry is a scalar multiple of a 2x2 block?
        # No, A_s is a 2x2 matrix. Its entries are 2x2 matrices.
        # So A_s[0,0] is a 2x2 matrix.
        # In M1, A_s[0,0] = A11 + A22. This is a 2x2 matrix.
        # So we need to represent A_s[i,j] as a linear combination of the 4 blocks.
        
        # Let's define A_s as a 2x2 matrix of coefficients for the 4 blocks.
        # A_s[i,j] is a 2x2 matrix.
        # We can represent A_s[i,j] by a 4x4 matrix of coefficients for the global A entries?
        # Or simpler: A_s[i,j] is a linear combination of the 4 blocks.
        # Let's use a dictionary or a 2x2 list of lists of (coeff, block_id).
        
        # For M1: A_1 = [[A11+A22, 0], [0, 0]]? No.
        # The multiplication is (A11+A22) * (B11+B22).
        # This is a product of two 2x2 matrices.
        # So the "A matrix" for this multiplication is just the 2x2 matrix A11+A22.
        # It is not a 2x2 matrix of 2x2 matrices.
        # It is a single 2x2 matrix.
        # So for each of the 7 outer steps, we have ONE 2x2 matrix A_s and ONE 2x2 matrix B_s.
        # We compute M_s = A_s * B_s.
        # A_s is a 2x2 matrix. Its entries are 2x2 matrices? No.
        # A_s is a 2x2 matrix of scalars? No.
        # A_s is a 2x2 matrix.
        # In the context of 4x4 multiplication, A is a 2x2 matrix of 2x2 blocks.
        # A11, A12, A21, A22 are 2x2 blocks.
        # A11+A22 is a 2x2 matrix.
        # So A_s is a 2x2 matrix.
        # We need to express A_s as a linear combination of the entries of A.
        # Since A_s is 2x2, and A is 4x4, A_s is formed by taking linear combinations of the 2x2 blocks.
        # So A_s[0,0] (top-left of A_s) is a linear combination of A11, A12, A21, A22?
        # No, A_s is a 2x2 matrix.
        # For M1, A_s = A11 + A22. This is a 2x2 matrix.
        # So A_s[0,0] = A11[0,0] + A22[0,0].
        # A_s[0,1] = A11[0,1] + A22[0,1].
        # etc.
        
        # So for each outer step s, we define A_s as a 2x2 matrix where each entry is a linear combination of the 4 blocks.
        # We can represent this as a 2x2 matrix of lists of (coeff, block_id).
        
        # M1: A_s = [[A11+A22, 0], [0, 0]]? No.
        # The product is (A11+A22)(B11+B22).
        # So A_s is the 2x2 matrix A11+A22.
        # B_s is the 2x2 matrix B11+B22.
        # So A_s[0,0] = A11[0,0]+A22[0,0], etc.
        # This means A_s is formed by adding the blocks element-wise.
        
        # Let's define the 7 combinations:
        # 1. A_s = A11+A22, B_s = B11+B22
        # 2. A_s = A21+A22, B_s = B11
        # 3. A_s = A11, B_s = B12-B22
        # 4. A_s = A22, B_s = B21-B11
        # 5. A_s = A11+A12, B_s = B22
        # 6. A_s = A21-A11, B_s = B11+B12
        # 7. A_s = A12-A22, B_s = B21+B22
        
        # For each, A_s is a 2x2 matrix.
        # We need to map the entries of A_s to the global A indices.
        # A_s[i,j] is a linear combination of the 4 blocks.
        # Let's define A_s_coeffs[i][j] as a list of (coeff, block_id).
        
    ]
    
    # This is getting complex to hardcode. Let's write a helper.
    
    def make_block_combo(coeffs):
        # coeffs is a list of (coeff, block_id)
        # Returns a 2x2 matrix of lists of (coeff, global_i, global_j)
        # Actually, we just need to accumulate into U, V, W.
        pass

    # Let's just implement the 49 terms directly by iterating s and t.
    
    # Define the 7 outer steps
    # Each step is (A_blocks, B_blocks)
    # A_blocks is a 2x2 matrix of lists of (coeff, block_id)
    
    def get_A_s(s):
        # Returns 2x2 matrix of lists of (coeff, block_id)
        if s == 0:
            # A11 + A22
            return [
                [[(1, 0), (1, 3)]],
                [[(1, 0), (1, 3)]]
            ] # This is wrong, A_s is 2x2.
            # A_s[0,0] = A11[0,0] + A22[0,0]
            # A_s[0,1] = A11[0,1] + A22[0,1]
            # A_s[1,0] = A11[1,0] + A22[1,0]
            # A_s[1,1] = A11[1,1] + A22[1,1]
            # So for each entry (i,j) of A_s, the combo is the same.
            return [
                [[(1, 0), (1, 3)], [(1, 0), (1, 3)]],
                [[(1, 0), (1, 3)], [(1, 0), (1, 3)]]
            ]
        elif s == 1:
            # A21 + A22
            return [
                [[(1, 2), (1, 3)], [(1, 2), (1, 3)]],
                [[(1, 2), (1, 3)], [(1, 2), (1, 3)]]
            ]
        elif s == 2:
            # A11
            return [
                [[(1, 0)], [(1, 0)]],
                [[(1, 0)], [(1, 0)]]
            ]
        elif s == 3:
            # A22
            return [
                [[(1, 3)], [(1, 3)]],
                [[(1, 3)], [(1, 3)]]
            ]
        elif s == 4:
            # A11 + A12
            return [
                [[(1, 0), (1, 1)], [(1, 0), (1, 1)]],
                [[(1, 0), (1, 1)], [(1, 0), (1, 1)]]
            ]
        elif s == 5:
            # A21 - A11
            return [
                [[(1, 2), (-1, 0)], [(1, 2), (-1, 0)]],
                [[(1, 2), (-1, 0)], [(1, 2), (-1, 0)]]
            ]
        elif s == 6:
            # A12 - A22
            return [
                [[(1, 1), (-1, 3)], [(1, 1), (-1, 3)]],
                [[(1, 1), (-1, 3)], [(1, 1), (-1, 3)]]
            ]

    def get_B_s(s):
        if s == 0:
            # B11 + B22
            return [
                [[(1, 0), (1, 3)], [(1, 0), (1, 3)]],
                [[(1, 0), (1, 3)], [(1, 0), (1, 3)]]
            ]
        elif s == 1:
            # B11
            return [
                [[(1, 0)], [(1, 0)]],
                [[(1, 0)], [(1, 0)]]
            ]
        elif s == 2:
            # B12 - B22
            return [
                [[(1, 1), (-1, 3)], [(1, 1), (-1, 3)]],
                [[(1, 1), (-1, 3)], [(1, 1), (-1, 3)]]
            ]
        elif s == 3:
            # B21 - B11
            return [
                [[(1, 2), (-1, 0)], [(1, 2), (-1, 0)]],
                [[(1, 2), (-1, 0)], [(1, 2), (-1, 0)]]
            ]
        elif s == 4:
            # B22
            return [
                [[(1, 3)], [(1, 3)]],
                [[(1, 3)], [(1, 3)]]
            ]
        elif s == 5:
            # B11 + B12
            return [
                [[(1, 0), (1, 1)], [(1, 0), (1, 1)]],
                [[(1, 0), (1, 1)], [(1, 0), (1, 1)]]
            ]
        elif s == 6:
            # B21 + B22
            return [
                [[(1, 2), (1, 3)], [(1, 2), (1, 3)]],
                [[(1, 2), (1, 3)], [(1, 2), (1, 3)]]
            ]

    # Now, for each s, we have A_s and B_s.
    # We compute the 7 Strassen terms for A_s * B_s.
    # The terms are bilinear forms in the entries of A_s and B_s.
    # We need to expand these into global A and B entries.
    
    # Let's define the 7 Strassen terms for 2x2.
    # t=0: (A00+A11)(B00+B11)
    # t=1: (A10+A11)B00
    # t=2: A00(B01-B11)
    # t=3: A11(B10-B00)
    # t=4: (A00+A01)B11
    # t=5: (A10-A00)(B00+B01)
    # t=6: (A01-A11)(B10+B11)
    
    # For each term t, we have a linear combination of A_s entries and B_s entries.
    # Let A_combo be a 2x2 matrix of lists of (coeff, local_i, local_j).
    # Let B_combo be a 2x2 matrix of lists of (coeff, local_i, local_j).
    # Then the term is sum_{i,j,k,l} A_combo[i,j] * B_combo[k,l] * A_s[i,j] * B_s[k,l]?
    # No, the term is a product of two linear combinations.
    # L_A = sum_{i,j} c_A[i,j] A_s[i,j]
    # L_B = sum_{k,l} c_B[k,l] B_s[k,l]
    # Term = L_A * L_B = sum_{i,j,k,l} c_A[i,j] c_B[k,l] A_s[i,j] B_s[k,l]
    
    # A_s[i,j] is a linear combination of global A entries.
    # A_s[i,j] = sum_{(r,c) in global} a_{s,i,j,r,c} A[r,c]
    # Similarly for B_s.
    
    # So Term = sum_{i,j,k,l} c_A[i,j] c_B[k,l] (sum_{r,c} a_{s,i,j,r,c} A[r,c]) (sum_{r',c'} b_{s,k,l,r',c'} B[r',c'])
    # = sum_{r,c,r',c'} (sum_{i,j,k,l} c_A[i,j] c_B[k,l] a_{s,i,j,r,c} b_{s,k,l,r',c'}) A[r,c] B[r',c']
    
    # We can compute the coefficients for U and V directly.
    
    # Let's precompute the expansion of A_s and B_s.
    # A_s_expanded[s][i][j] is a dictionary { (r,c): coeff }
    
    A_s_expanded = []
    for s in range(7):
        A_s_def = get_A_s(s)
        A_s_exp = [[{} for _ in range(2)] for _ in range(2)]
        for i in range(2):
            for j in range(2):
                for coeff, block_id in A_s_def[i][j]:
                    # Expand block_id to global indices
                    # Block 0: 11, 1: 12, 2: 21, 3: 22
                    # Local (0,0) -> (0,0), (0,1) -> (0,1), etc.
                    # But A_s[i,j] is the (i,j) entry of the 2x2 matrix A_s.
                    # The block itself is 2x2.
                    # So A_s[i,j] = sum_{local_u, local_v} coeff * Block[block_id][local_u, local_v]
                    # Wait, A_s[i,j] is a scalar? No, A_s is a 2x2 matrix of 2x2 blocks?
                    # No, A_s is a 2x2 matrix.
                    # In the context of Strassen for 4x4, A_s is a 2x2 matrix.
                    # Its entries are scalars? No, they are 2x2 matrices?
                    # No, A_s is a 2x2 matrix of scalars?
                    # Let's re-read Strassen.
                    # For 2x2, A is 2x2.
                    # For 4x4, A is 2x2 of 2x2 blocks.
                    # The "A_s" in the outer step is a 2x2 matrix of 2x2 blocks?
                    # No, the outer step computes M_s = A_s B_s.
                    # A_s and B_s are 2x2 matrices.
                    # But they are formed from the 2x2 blocks of A and B.
                    # So A_s is a 2x2 matrix, where each entry is a 2x2 matrix?
                    # No, if A_s is a 2x2 matrix, then A_s[0,0] is a scalar.
                    # But A11+A22 is a 2x2 matrix.
                    # So A_s is a 2x2 matrix of 2x2 matrices?
                    # No, that would be 4x4.
                    # The product M_s is a 2x2 matrix of 2x2 matrices?
                    # No, M_s is a 2x2 matrix.
                    # So A_s and B_s must be 2x2 matrices of scalars?
                    # No, that doesn't make sense.
                    
                    # Let's clarify.
                    # A is 4x4.
                    # A = [[A11, A12], [A21, A22]] where Aij are 2x2.
                    # M1 = (A11+A22)(B11+B22).
                    # A11+A22 is a 2x2 matrix.
                    # B11+B22 is a 2x2 matrix.
                    # Their product is a 2x2 matrix.
                    # So A_s = A11+A22 (2x2 matrix).
                    # B_s = B11+B22 (2x2 matrix).
                    # M_s = A_s B_s (2x2 matrix).
                    # So A_s is a 2x2 matrix.
                    # Its entries are scalars.
                    # A_s[0,0] = A11[0,0] + A22[0,0].
                    # A_s[0,1] = A11[0,1] + A22[0,1].
                    # etc.
                    
                    # So for each entry (i,j) of A_s, we have a linear combination of the 4 blocks.
                    # And each block is a 2x2 matrix.
                    # So A_s[i,j] = sum_{block} coeff * Block[block][i,j]?
                    # No, Block[block] is a 2x2 matrix.
                    # A_s[i,j] is a scalar.
                    # A_s[i,j] = sum_{block} coeff * Block[block][i,j].
                    # This assumes that the combination is applied element-wise.
                    # Yes, A11+A22 means element-wise addition.
                    
                    # So for a fixed s, and fixed i,j (local 2x2 index):
                    # A_s[i,j] = sum_{block} coeff_{s,i,j,block} * A_global[block][i,j]
                    # Where A_global[block][i,j] is the entry at local index (i,j) of the block.
                    # And A_global[block][i,j] maps to a specific global index (r,c).
                    
                    # So we can compute the coefficient for each global (r,c).
                    
                    for li in range(2):
                        for lj in range(2):
                            # Get global index for block_id, li, lj
                            # Block 0 (11): (0,0), (0,1), (1,0), (1,1)
                            # Block 1 (12): (0,2), (0,3), (1,2), (1,3)
                            # Block 2 (21): (2,0), (2,1), (3,0), (3,1)
                            # Block 3 (22): (2,2), (2,3), (3,2), (3,3)
                            # li, lj are 0 or 1.
                            # Global row = (block_row * 2) + li
                            # Global col = (block_col * 2) + lj
                            # Block 0: row 0, col 0
                            # Block 1: row 0, col 1
                            # Block 2: row 1, col 0
                            # Block 3: row 1, col 1
                            br = block_id // 2
                            bc = block_id % 2
                            r = br * 2 + li
                            c = bc * 2 + lj
                            # Add to A_s_exp[s][i][j][(r,c)]
                            if (r,c) in A_s_exp[i][j]:
                                A_s_exp[i][j][(r,c)] += coeff
                            else:
                                A_s_exp[i][j][(r,c)] = coeff
        A_s_expanded.append(A_s_exp)
        
    # Similarly for B_s_expanded.
    B_s_expanded = []
    for s in range(7):
        B_s_def = get_B_s(s)
        B_s_exp = [[{} for _ in range(2)] for _ in range(2)]
        for i in range(2):
            for j in range(2):
                for coeff, block_id in B_s_def[i][j]:
                    for li in range(2):
                        for lj in range(2):
                            br = block_id // 2
                            bc = block_id % 2
                            r = br * 2 + li
                            c = bc * 2 + lj
                            # B is 4x4. Indices (r,c).
                            # V is indexed by j*p + k. Here j=r, k=c.
                            # But we only use the first 4 columns of B (k=0..3).
                            if c < 4:
                                v_idx = r * 7 + c
                                if v_idx in B_s_exp[i][j]:
                                    B_s_exp[i][j][v_idx] += coeff
                                else:
                                    B_s_exp[i][j][v_idx] = coeff
        B_s_expanded.append(B_s_exp)

    # Now, for each s and t, we compute the term.
    # t=0..6.
    # Define the linear combinations for A_s and B_s for each t.
    
    # t=0: (A00+A11)(B00+B11)
    # L_A = A_s[0,0] + A_s[1,1]
    # L_B = B_s[0,0] + B_s[1,1]
    
    # We can define L_A and L_B as 2x2 matrices of coefficients for A_s and B_s.
    # L_A[i,j] is the coeff for A_s[i,j].
    
    L_A_defs = [
        [[1, 0], [0, 1]], # t=0
        [[0, 0], [1, 1]], # t=1: (A10+A11) -> A10 is [1,0], A11 is [1,1]
        [[1, 0], [0, 0]], # t=2: A00
        [[0, 0], [0, 1]], # t=3: A11
        [[1, 1], [0, 0]], # t=4: (A00+A01)
        [[0, 0], [1, -1]], # t=5: (A10-A00)
        [[0, 1], [0, -1]]  # t=6: (A01-A11)
    ]
    
    L_B_defs = [
        [[1, 0], [0, 1]], # t=0
        [[1, 0], [0, 0]], # t=1: B00
        [[0, 1], [0, -1]], # t=2: (B01-B11)
        [[0, 0], [1, -1]], # t=3: (B10-B00)
        [[0, 0], [0, 1]], # t=4: B11
        [[1, 1], [0, 0]], # t=5: (B00+B01)
        [[0, 0], [1, 1]]  # t=6: (B10+B11)
    ]
    
    # Output coefficients for C.
    # C00 = M1 - M4 + M6 - M7
    # C01 = M2 + M3
    # C10 = M4 + M5
    # C11 = M1 - M2 + M5 + M7
    
    # M_s is a 2x2 matrix.
    # M_s[i,j] is the (i,j) entry.
    # C_block[i,j] = sum_s d_{s,i,j} M_s[i,j]
    
    # d_{s,i,j} is the coefficient for M_s[i,j] in C_block[i,j].
    
    # s=0 (M1): C00 += 1, C11 += 1
    # s=1 (M2): C01 += 1, C11 -= 1
    # s=2 (M3): C01 += 1
    # s=3 (M4): C00 -= 1, C10 += 1
    # s=4 (M5): C10 += 1, C11 += 1
    # s=5 (M6): C00 += 1
    # s=6 (M7): C00 -= 1, C11 += 1
    
    C_defs = [
        # s=0
        [[1, 0], [0, 1]],
        # s=1
        [[0, 1], [0, -1]],
        # s=2
        [[0, 1], [0, 0]],
        # s=3
        [[-1, 0], [1, 0]],
        # s=4
        [[0, 0], [1, 1]],
        # s=5
        [[1, 0], [0, 0]],
        # s=6
        [[-1, 0], [0, 1]]
    ]
    
    col = 0
    for s in range(7):
        for t in range(7):
            # Compute U, V, W for this term
            # U: coeffs for A
            # V: coeffs for B
            # W: coeffs for C
            
            # U[r,c] = sum_{i,j} L_A_defs[t][i][j] * A_s_expanded[s][i][j][(r,c)]
            # V[r,c] = sum_{i,j} L_B_defs[t][i][j] * B_s_expanded[s][i][j][(r,c)]
            # W[r,c] = sum_{i,j} C_defs[s][i][j] * (coeff for M_s[i,j])
            # Wait, W is for the output C.
            # The term is L_A * L_B.
            # This contributes to M_s[i,j]?
            # No, the Strassen term m_{s,t} is a scalar.
            # m_{s,t} is used in the linear combination for C.
            # m_{s,t} = L_A * L_B.
            # C_block[i,j] = sum_t d_{s,t,i,j} m_{s,t}
            # So W[r,c] (for C) = sum_{i,j} d_{s,t,i,j} * (coeff for C_block[i,j])
            # But C_block[i,j] is a 2x2 block.
            # C_block[i,j][u,v] is the entry at local (u,v).
            # Global C index is (r,c).
            # C_block[i,j][u,v] maps to global (r,c).
            
            # So for a fixed s, t:
            # m_{s,t} = L_A * L_B.
            # This scalar m_{s,t} is added to C_block[i,j] with coefficient d_{s,t,i,j}.
            # So C_block[i,j] += d_{s,t,i,j} * m_{s,t}.
            # C_block[i,j][u,v] += d_{s,t,i,j} * m_{s,t}.
            # So W[global_C_index] += d_{s,t,i,j} * 1 (since m_{s,t} is the variable).
            # Wait, m_{s,t} is a product of A and B entries.
            # So in the tensor decomposition, the term is:
            # U[A_index] * V[B_index] * W[C_index]
            # The coefficient for W[C_index] is d_{s,t,i,j} where C_index corresponds to C_block[i,j][u,v].
            
            # Let's compute U and V first.
            U_term = [0] * 16
            V_term = [0] * 28
            
            for i in range(2):
                for j in range(2):
                    coeff_A = L_A_defs[t][i][j]
                    if coeff_A != 0:
                        for (r,c), coeff in A_s_expanded[s][i][j].items():
                            u_idx = r * 4 + c
                            U_term[u_idx] += coeff_A * coeff
                            
                    coeff_B = L_B_defs[t][i][j]
                    if coeff_B != 0:
                        for v_idx, coeff in B_s_expanded[s][i][j].items():
                            V_term[v_idx] += coeff_B * coeff
            
            # Now W.
            W_term = [0] * 28
            for i in range(2):
                for j in range(2):
                    coeff_C = C_defs[s][i][j]
                    if coeff_C != 0:
                        # C_block[i,j] is a 2x2 block.
                        # Local (u,v) in C_block[i,j] maps to global (r,c).
                        # br = i, bc = j.
                        # r = br * 2 + u, c = bc * 2 + v.
                        for u in range(2):
                            for v in range(2):
                                r = i * 2 + u
                                c = j * 2 + v
                                w_idx = c * 4 + r # W is indexed by k*n + i. Here k=c, i=r.
                                W_term[w_idx] += coeff_C
            
            # Store in U, V, W
            for r in range(16):
                U[r][col] = U_term[r]
            for r in range(28):
                V[r][col] = V_term[r]
                W[r][col] = W_term[r]
                
            col += 1
            
    # Now add the schoolbook part for 4x4x3
    # B_2 is 4x3. C_2 is 4x3.
    # C_2[i,k] = sum_j A[i,j] B_2[j,k]
    # B_2[j,k] corresponds to B[j, 4+k].
    # C_2[i,k] corresponds to C[i, 4+k].
    
    for i in range(4):
        for j in range(4):
            for k in range(3):
                u_idx = i * 4 + j
                v_idx = j * 7 + (4 + k)
                w_idx = (4 + k) * 4 + i
                
                U[u_idx][col] = 1
                V[v_idx][col] = 1
                W[w_idx][col] = 1
                col += 1
                
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
