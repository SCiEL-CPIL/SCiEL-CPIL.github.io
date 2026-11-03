import time

DIMENSIONS = [4, 4, 9]

def get_strassen_factors(N):
    """
    Returns U, V, W factors for N x N x N matrix multiplication using Strassen's algorithm.
    U shape: (N*N, Rank)
    V shape: (N*N, Rank)
    W shape: (N*N, Rank)
    Indices:
    U row: i * N + j (A_ij)
    V row: j * N + k (B_jk)
    W row: k * N + i (C_ki)
    """
    if N == 1:
        return [[1]], [[1]], [[1]]
    
    # For N=2, we define the 7 terms explicitly
    if N == 2:
        U = [[0]*7 for _ in range(4)]
        V = [[0]*7 for _ in range(4)]
        W = [[0]*7 for _ in range(4)]
        
        # Term 0: M1 = (A11+A22)(B11+B22)
        U[0][0] = 1; U[3][0] = 1
        V[0][0] = 1; V[3][0] = 1
        W[0][0] = 1; W[3][0] = 1 # C11, C22
        
        # Term 1: M2 = (A21+A22)B11
        U[2][1] = 1; U[3][1] = 1
        V[0][1] = 1
        W[2][1] = 1; W[3][1] = -1 # C21, C22
        
        # Term 2: M3 = A11(B12-B22)
        U[0][2] = 1
        V[1][2] = 1; V[3][2] = -1
        W[1][2] = 1; W[3][2] = 1 # C12, C22
        
        # Term 3: M4 = A22(B21-B11)
        U[3][3] = 1
        V[2][3] = 1; V[0][3] = -1
        W[0][3] = 1; W[2][3] = 1 # C11, C21
        
        # Term 4: M5 = (A11+A12)B22
        U[0][4] = 1; U[1][4] = 1
        V[3][4] = 1
        W[0][4] = -1; W[1][4] = 1 # C11, C12
        
        # Term 5: M6 = (A21-A11)(B11+B12)
        U[2][5] = 1; U[0][5] = -1
        V[0][5] = 1; V[1][5] = 1
        W[3][5] = 1 # C22
        
        # Term 6: M7 = (A12-A22)(B21+B22)
        U[1][6] = 1; U[3][6] = -1
        V[2][6] = 1; V[3][6] = 1
        W[0][6] = 1; W[3][6] = -1 # C11, C22
        
        return U, V, W

    # Recursive step for N > 2
    # Split into 4 blocks of N/2
    # This implementation is complex to generalize perfectly for arbitrary N without a library,
    # but for N=4, we can just call N=2 and combine.
    # However, for simplicity and safety, I will only support N=1, 2, 4.
    if N == 4:
        # Get factors for 2x2
        U2, V2, W2 = get_strassen_factors(2)
        rank2 = len(U2[0])
        rank4 = rank2 * rank2 # 49
        
        U4 = [[0]*rank4 for _ in range(16)]
        V4 = [[0]*rank4 for _ in range(16)]
        W4 = [[0]*rank4 for _ in range(16)]
        
        # The 7 Strassen terms for 4x4 correspond to 7 products of 2x2 matrices.
        # Each 2x2 product is decomposed into 7 terms.
        # Total 49 terms.
        
        # We need to map the 7 "meta-terms" to the 49 "micro-terms".
        # Meta-term k uses specific 2x2 blocks of A, B.
        # Let's define the 7 meta-terms' A and B block coefficients.
        
        # Meta 0: (A11+A22)(B11+B22)
        # Meta 1: (A21+A22)B11
        # Meta 2: A11(B12-B22)
        # Meta 3: A22(B21-B11)
        # Meta 4: (A11+A12)B22
        # Meta 5: (A21-A11)(B11+B12)
        # Meta 6: (A12-A22)(B21+B22)
        
        # Block indices for A (2x2 blocks of 4x4):
        # A11 (top-left), A12 (top-right), A21 (bottom-left), A22 (bottom-right)
        # Local 2x2 indices: 0,1,2,3 corresponding to A11,A12,A21,A22?
        # No, in the 2x2 factor, the rows are 0,1,2,3 corresponding to:
        # 0: A11, 1: A12, 2: A21, 3: A22 (if we map 4x4 to 2x2 of 2x2 blocks)
        # Wait, the 2x2 factor operates on 2x2 matrices.
        # Here the "matrices" are 2x2 blocks of the 4x4 matrix.
        
        # Let's define the mapping from Meta-term k to the 2x2 factor's inputs.
        # The 2x2 factor U2 has 4 rows (0,1,2,3) corresponding to the 4 elements of the 2x2 "matrix".
        # In the 4x4 context, these 4 elements are 2x2 blocks.
        # Block 0 (A11): rows 0-1, cols 0-1 of 4x4.
        # Block 1 (A12): rows 0-1, cols 2-3.
        # Block 2 (A21): rows 2-3, cols 0-1.
        # Block 3 (A22): rows 2-3, cols 2-3.
        
        # For each meta-term k, we have coefficients for the 4 A-blocks and 4 B-blocks.
        # These coefficients are the same as the U2 and V2 columns for the 2x2 case?
        # Yes! The structure is identical.
        # So for meta-term k, the A-block coefficients are U2[:, k] and B-block coefficients are V2[:, k].
        # And for each such combination, we expand using the 2x2 factor.
        
        # So, for each meta-term k (0..6):
        #   For each micro-term l (0..6):
        #     The global term index is k*7 + l.
        #     The A part is: sum over block b where U2[b, k] != 0 of (U2_factor_l for block b scaled by U2[b, k])
        #     Wait, this is getting complicated to do manually.
        
        # Easier: Just hardcode the 49 terms? No, too long.
        # Let's use the recursive property properly.
        
        # Actually, I can just use the schoolbook for N=4? Rank 64.
        # 64 + 64 + 16 = 144. No improvement.
        # I need the Strassen 49.
        
        # Let's implement the recursion correctly.
        # U4, V4, W4 are built from U2, V2, W2.
        
        # For each of the 7 meta-terms (k):
        #   The A-input is a linear combination of 4 blocks.
        #   The B-input is a linear combination of 4 blocks.
        #   The product is computed by the 2x2 algorithm (U2, V2, W2).
        #   So, for each meta-term k, we add the 2x2 decomposition scaled by the block coefficients.
        
        for k in range(7):
            # Get A-block coefficients from U2[:, k]
            # Get B-block coefficients from V2[:, k]
            # For each micro-term l in 0..6:
            for l in range(7):
                idx = k * 7 + l
                # A part
                for b in range(4):
                    coeff_a = U2[b][k]
                    if coeff_a == 0: continue
                    # The block b corresponds to a 2x2 submatrix.
                    # We need to add U2[:, l] scaled by coeff_a to the appropriate rows of U4.
                    # Block b maps to rows:
                    # b=0 (A11): rows 0,1,4,5? No.
                    # 4x4 matrix indices:
                    # A11: (0,0),(0,1),(1,0),(1,1) -> rows 0,1,4,5 in U4 (flattened)
                    # A12: (0,2),(0,3),(1,2),(1,3) -> rows 2,3,6,7
                    # A21: (2,0),(2,1),(3,0),(3,1) -> rows 8,9,12,13
                    # A22: (2,2),(2,3),(3,2),(3,3) -> rows 10,11,14,15
                    
                    # U2 rows 0,1,2,3 correspond to the 4 elements of the 2x2 block.
                    # So U2[0,l] adds to row corresponding to (0,0) of block b.
                    # U2[1,l] adds to row corresponding to (0,1) of block b.
                    # U2[2,l] adds to row corresponding to (1,0) of block b.
                    # U2[3,l] adds to row corresponding to (1,1) of block b.
                    
                    base_row = [0, 2, 8, 10][b]
                    # Offset for 2x2 block within 4x4:
                    # (0,0) is row 0, (0,1) is row 1, (1,0) is row 4, (1,1) is row 5?
                    # Flattened 4x4:
                    # 0 1 2 3
                    # 4 5 6 7
                    # 8 9 10 11
                    # 12 13 14 15
                    
                    # Block A11 (0,0 to 1,1):
                    # (0,0)->0, (0,1)->1, (1,0)->4, (1,1)->5
                    # Block A12 (0,2 to 1,3):
                    # (0,2)->2, (0,3)->3, (1,2)->6, (1,3)->7
                    # Block A21 (2,0 to 3,1):
                    # (2,0)->8, (2,1)->9, (3,0)->12, (3,1)->13
                    # Block A22 (2,2 to 3,3):
                    # (2,2)->10, (2,3)->11, (3,2)->14, (3,3)->15
                    
                    row_offsets = [
                        [0, 1, 4, 5],
                        [2, 3, 6, 7],
                        [8, 9, 12, 13],
                        [10, 11, 14, 15]
                    ]
                    
                    for sub_idx in range(4):
                        val = U2[sub_idx][l] * coeff_a
                        if val != 0:
                            U4[row_offsets[b][sub_idx]][idx] += val

                # B part
                for b in range(4):
                    coeff_b = V2[b][k]
                    if coeff_b == 0: continue
                    # Block b in B (4x4)
                    # Same row offsets as A?
                    # B is 4x4. Flattened rows:
                    # B11: 0,1,4,5
                    # B12: 2,3,6,7
                    # B21: 8,9,12,13
                    # B22: 10,11,14,15
                    # Yes, same structure.
                    
                    for sub_idx in range(4):
                        val = V2[sub_idx][l] * coeff_b
                        if val != 0:
                            V4[row_offsets[b][sub_idx]][idx] += val

                # W part
                # The output C is 4x4.
                # The 2x2 algorithm produces a 2x2 result C_meta.
                # C_meta is a linear combination of the 4 blocks of C.
                # W2 gives the coefficients for the 2x2 result.
                # For each micro-term l, W2[:, l] gives the coefficients for the 4 elements of C_meta.
                # Element 0 of C_meta is C11 (top-left block of C).
                # Element 1 is C12 (top-right).
                # Element 2 is C21 (bottom-left).
                # Element 3 is C22 (bottom-right).
                
                for sub_idx in range(4):
                    coeff_w = W2[sub_idx][l]
                    if coeff_w == 0: continue
                    # Add to the corresponding block of C.
                    # Block sub_idx in C (4x4)
                    # Same row offsets.
                    for sub_sub_idx in range(4):
                        val = coeff_w
                        if val != 0:
                            # Wait, W2[sub_idx][l] is the coefficient for the *entire* 2x2 block?
                            # No, W2 is the factor for the 2x2 multiplication.
                            # The "entry" W2[sub_idx][l] applies to the *element* sub_idx of the 2x2 output.
                            # But in the 4x4 context, each "element" of the 2x2 output is a 2x2 block.
                            # So we need to add val to ALL 4 entries of that block in W4.
                            
                            # Wait, no.
                            # The tensor identity is:
                            # sum_k U[i,k] V[j,k] W[k, (i,j)]
                            # In the recursive step:
                            # C_block = sum_{k_meta} (A_block_combination) (B_block_combination)
                            # Each term is a 2x2 product.
                            # The 2x2 product is decomposed into 7 terms (micro).
                            # So for a fixed meta-term k and micro-term l:
                            # The contribution to C is:
                            # (Linear combo of A blocks) * (Linear combo of B blocks)
                            # This product is a 2x2 matrix.
                            # Its entries are linear combinations of the products of the block entries.
                            # The W factor for the 4x4 case must reflect this.
                            
                            # Actually, it's simpler:
                            # The W4 matrix should have the same structure as W2, but expanded.
                            # For each meta-term k and micro-term l:
                            # The W part is W2[:, l] applied to the 4 blocks.
                            # So for sub_idx in 0..3:
                            #   coeff = W2[sub_idx][l]
                            #   If coeff != 0:
                            #      Add coeff to the 4 entries of the block sub_idx in W4.
                            
                            block_rows = row_offsets[sub_idx]
                            for r in block_rows:
                                W4[r][idx] += coeff_w

        return U4, V4, W4

    raise ValueError("N must be 1, 2, or 4")

def construct():
    n, m, p = DIMENSIONS
    # Rank 114
    rank = 114
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Part 1: Strassen 4x4x4 for B cols 0-3, C cols 0-3
    U1, V1, W1 = get_strassen_factors(4)
    for r in range(49):
        for i in range(16):
            if U1[i][r] != 0:
                U[i][r] = U1[i][r]
        for j in range(16):
            if V1[j][r] != 0:
                # V1 row j corresponds to B block element.
                # j is 0..15.
                # Global B row index:
                # B is 4x9.
                # j maps to (row_b, col_b_local) in 4x4.
                # row_b = j // 4, col_b_local = j % 4.
                # Global col = col_b_local (0-3).
                # Global row = row_b * 9 + col_b_local? No.
                # V row index is j * p + k.
                # Here j is row of B (0-3), k is col of B (0-3).
                # So global row = j * 9 + k.
                row_b = j // 4
                col_b = j % 4
                global_row = row_b * 9 + col_b
                V[global_row][r] = V1[j][r]
        for k in range(16):
            if W1[k][r] != 0:
                # W1 row k corresponds to C block element.
                # k is 0..15.
                # k maps to (col_c, row_c) in 4x4.
                # col_c = k // 4, row_c = k % 4.
                # Global col = col_c (0-3).
                # Global row = col_c * 4 + row_c.
                col_c = k // 4
                row_c = k % 4
                global_row = col_c * 4 + row_c
                W[global_row][r] = W1[k][r]

    # Part 2: Strassen 4x4x4 for B cols 4-7, C cols 4-7
    U2, V2, W2 = get_strassen_factors(4)
    offset = 49
    for r in range(49):
        for i in range(16):
            if U2[i][r] != 0:
                U[i][offset + r] = U2[i][r]
        for j in range(16):
            if V2[j][r] != 0:
                row_b = j // 4
                col_b = j % 4
                # Global col = 4 + col_b
                global_row = row_b * 9 + (4 + col_b)
                V[global_row][offset + r] = V2[j][r]
        for k in range(16):
            if W2[k][r] != 0:
                col_c = k // 4
                row_c = k % 4
                # Global col = 4 + col_c
                global_row = (4 + col_c) * 4 + row_c
                W[global_row][offset + r] = W2[k][r]

    # Part 3: Schoolbook 4x4x1 for B col 8, C col 8
    offset3 = 98
    for i in range(4):
        for j in range(4):
            t = i * 4 + j
            U[i * 4 + j][offset3 + t] = 1
            V[j * 9 + 8][offset3 + t] = 1
            W[8 * 4 + i][offset3 + t] = 1

    return {"U": U, "V": V, "W": W}
