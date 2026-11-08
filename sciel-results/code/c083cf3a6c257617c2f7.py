"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    """
    2-level Strassen decomposition for 4x4 matrix multiplication.
    Rank 49 (known optimal for 4x4).
    
    The 4x4 matrices are partitioned into 2x2 blocks:
    A = [[A11, A12], [A21, A22]]  (each 2x2)
    B = [[B11, B12], [B21, B22]]  (each 2x2)
    C = [[C11, C12], [C21, C22]]  (each 2x2)
    
    Using Strassen at the top level gives 7 products of 2x2 blocks.
    Each 2x2 product uses Strassen with 7 scalar multiplications.
    Total: 7 * 7 = 49 scalar multiplications.
    """
    n, m, p = DIMENSIONS  # 4, 4, 4
    rank = 49
    
    U = [[0] * rank for _ in range(n * m)]  # (16, 49)
    V = [[0] * rank for _ in range(m * p)]  # (16, 49)
    W = [[0] * rank for _ in range(p * n)]  # (16, 49)
    
    # Strassen formulas for 2x2:
    # M1 = (A11+A22)*(B11+B22)
    # M2 = (A21+A22)*B11
    # M3 = A11*(B12-B22)
    # M4 = A22*(B21-B11)
    # M5 = (A11+A12)*B22
    # M6 = (A21-A11)*(B11+B12)
    # M7 = (A12-A22)*(B21+B22)
    #
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    # For 4x4, we partition into 2x2 blocks.
    # Block (bi, bj) of A corresponds to rows 2*bi to 2*bi+1, cols 2*bj to 2*bj+1
    # So element (i, j) of A is in block (i//2, j//2), position (i%2, j%2)
    
    # Strassen at top level (4x4 -> 7 products of 2x2 blocks):
    # M1 = (A11+A22)*(B11+B22)
    # M2 = (A21+A22)*B11
    # M3 = A11*(B12-B22)
    # M4 = A22*(B21-B11)
    # M5 = (A11+A12)*B22
    # M6 = (A21-A11)*(B11+B12)
    # M7 = (A12-A22)*(B21+B22)
    #
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    # For each top-level Strassen product Mt, we need to define:
    # - Which elements of A contribute (with signs)
    # - Which elements of B contribute (with signs)
    # - Which elements of C receive the result (with signs)
    
    # Top-level Strassen: 7 products
    # Each product involves sums/differences of 2x2 blocks
    # Within each 2x2 block product, we use Strassen again (7 scalar mults)
    
    # Let's define the top-level Strassen structure first
    # For product Mt, A-side blocks and B-side blocks:
    
    # M1: A-side: A11 + A22, B-side: B11 + B22
    # M2: A-side: A21 + A22, B-side: B11
    # M3: A-side: A11, B-side: B12 - B22
    # M4: A-side: A22, B-side: B21 - B11
    # M5: A-side: A11 + A12, B-side: B22
    # M6: A-side: A21 - A11, B-side: B11 + B12
    # M7: A-side: A12 - A22, B-side: B21 + B22
    
    # For each top-level product, the A-side is a sum of blocks, B-side is a sum of blocks.
    # The product of two block sums is:
    # (sum of A_blocks) * (sum of B_blocks)
    # But Strassen at the bottom level handles the 2x2 block multiplication.
    
    # Actually, let me think about this differently.
    # The tensor decomposition has 49 terms.
    # Each term is a rank-1 tensor: u * v * w^T
    # where u is a vector of length 16 (for A), v is length 16 (for B), w is length 16 (for C)
    
    # Let me construct this more systematically.
    
    # Helper: for a 2x2 block at position (bi, bj), the four elements are at:
    # (2*bi + 0, 2*bj + 0), (2*bi + 0, 2*bj + 1)
    # (2*bi + 1, 2*bj + 0), (2*bi + 1, 2*bj + 1)
    
    # Strassen at the 2x2 level:
    # Given 2x2 matrices P and Q:
    # P = [[p00, p01], [p10, p11]]
    # Q = [[q00, q01], [q10, q11]]
    #
    # S1 = (p00 + p11) * (q00 + q11)
    # S2 = (p10 + p11) * q00
    # S3 = p00 * (q01 - q11)
    # S4 = p11 * (q10 - q00)
    # S5 = (p00 + p01) * q11
    # S6 = (p10 - p00) * (q00 + q01)
    # S7 = (p01 - p11) * (q10 + q11)
    #
    # R00 = S1 + S4 - S5 + S7
    # R01 = S3 + S5
    # R10 = S2 + S4
    # R11 = S1 - S2 + S3 + S6
    
    # For the 4x4 case with top-level Strassen:
    # We have 7 top-level products. Each top-level product is a product of two "block sums".
    # Each block sum is a linear combination of 2x2 blocks.
    # The product of two such block sums, when expanded, gives a 2x2 block result.
    # But we need to be careful: the top-level Strassen combines 2x2 block products.
    
    # Let me use a different approach. I'll directly construct the 49 rank-1 terms.
    
    # Top-level Strassen for 4x4:
    # A blocks: A11, A12, A21, A22 (each 2x2)
    # B blocks: B11, B12, B21, B22 (each 2x2)
    
    # M1 = (A11+A22) * (B11+B22)
    # M2 = (A21+A22) * B11
    # M3 = A11 * (B12-B22)
    # M4 = A22 * (B21-B11)
    # M5 = (A11+A12) * B22
    # M6 = (A21-A11) * (B11+B12)
    # M7 = (A12-A22) * (B21+B22)
    
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    # Now, each Mt is a 2x2 block. To compute Mt, we use Strassen on the 2x2 blocks.
    # But the "blocks" in Mt are not simple 2x2 matrices - they are linear combinations.
    
    # Actually, let me reconsider. The standard 2-level Strassen works as follows:
    # At the top level, we form 7 products of 2x2 blocks.
    # Each 2x2 block product is computed using Strassen (7 scalar multiplications).
    # Total: 7 * 7 = 49 scalar multiplications.
    
    # But the "2x2 blocks" in the top-level Strassen are linear combinations of the original 2x2 blocks.
    # For example, in M1 = (A11+A22) * (B11+B22), the left factor is (A11+A22), a 2x2 matrix.
    # The right factor is (B11+B22), a 2x2 matrix.
    # We compute this product using Strassen, which requires 7 scalar multiplications.
    
    # So for each top-level product Mt, we need to:
    # 1. Identify the 2x2 matrix on the A-side (sum/difference of original 2x2 blocks)
    # 2. Identify the 2x2 matrix on the B-side
    # 3. Apply Strassen to compute the 2x2 product
    # 4. Add the result to the appropriate C blocks
    
    # Let me define the top-level Strassen products more precisely.
    
    # For M1: A-side = A11 + A22, B-side = B11 + B22
    # A11 is at rows 0-1, cols 0-1
    # A22 is at rows 2-3, cols 2-3
    # So A-side = A11 + A22 means:
    #   (0,0): A[0,0] + A[2,2]
    #   (0,1): A[0,1] + A[2,3]
    #   (1,0): A[1,0] + A[3,2]
    #   (1,1): A[1,1] + A[3,3]
    #
    # B11 is at rows 0-1, cols 0-1
    # B22 is at rows 2-3, cols 2-3
    # So B-side = B11 + B22 means:
    #   (0,0): B[0,0] + B[2,2]
    #   (0,1): B[0,1] + B[2,3]
    #   (1,0): B[1,0] + B[3,2]
    #   (1,1): B[1,1] + B[3,3]
    
    # Now, to compute the 2x2 product of these two 2x2 matrices using Strassen,
    # we need 7 scalar multiplications. Each scalar multiplication involves:
    # - A linear combination of elements from the A-side 2x2 matrix
    # - A linear combination of elements from the B-side 2x2 matrix
    
    # Let me define a helper function to add a Strassen decomposition for a 2x2 product.
    
    def add_strassen_2x2(col_offset, a_blocks, b_blocks, c_blocks, c_signs):
        """
        Add Strassen decomposition for a 2x2 block product.
        
        a_blocks: list of (block_row, block_col, sign) for A-side
        b_blocks: list of (block_row, block_col, sign) for B-side
        c_blocks: list of (block_row, block_col, sign) for C-side
        c_signs: not used, c_blocks includes signs
        
        col_offset: starting column index
        """
        # Strassen formulas for 2x2:
        # S1 = (p00 + p11) * (q00 + q11)
        # S2 = (p10 + p11) * q00
        # S3 = p00 * (q01 - q11)
        # S4 = p11 * (q10 - q00)
        # S5 = (p00 + p01) * q11
        # S6 = (p10 - p00) * (q00 + q01)
        # S7 = (p01 - p11) * (q10 + q11)
        #
        # R00 = S1 + S4 - S5 + S7
        # R01 = S3 + S5
        # R10 = S2 + S4
        # R11 = S1 - S2 + S3 + S6
        
        # For each Strassen multiplication S_k, define:
        # - Which elements of the 2x2 A-side matrix contribute
        # - Which elements of the 2x2 B-side matrix contribute
        # - Which elements of the 2x2 C-side matrix receive the result
        
        # S1: (p00+p11) * (q00+q11)
        # S2: (p10+p11) * q00
        # S3: p00 * (q01-q11)
        # S4: p11 * (q10-q00)
        # S5: (p00+p01) * q11
        # S6: (p10-p00) * (q00+q01)
        # S7: (p01-p11) * (q10+q11)
        
        # For each S_k, the A-side is a linear combination of p00, p01, p10, p11
        # The B-side is a linear combination of q00, q01, q10, q11
        # The result contributes to R with certain signs
        
        # Let me define the Strassen multipliers:
        # Each entry is (a_coeffs, b_coeffs, c_coeffs)
        # a_coeffs: dict from (row, col) to coefficient for the 2x2 A matrix
        # b_coeffs: dict from (row, col) to coefficient for the 2x2 B matrix
        # c_coeffs: dict from (row, col) to coefficient for the 2x2 C matrix
        
        strassen_defs = [
            # S1: (p00+p11) * (q00+q11)
            ({(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1},
             {(0,0): 1, (1,1): 1}),  # R00 += S1, R11 += S1
            # S2: (p10+p11) * q00
            ({(1,0): 1, (1,1): 1}, {(0,0): 1},
             {(1,0): 1, (1,1): -1}),  # R10 += S2, R11 -= S2
            # S3: p00 * (q01-q11)
            ({(0,0): 1}, {(0,1): 1, (1,1): -1},
             {(0,1): 1, (1,1): 1}),  # R01 += S3, R11 += S3
            # S4: p11 * (q10-q00)
            ({(1,1): 1}, {(1,0): 1, (0,0): -1},
             {(0,0): 1, (1,0): 1}),  # R00 += S4, R10 += S4
            # S5: (p00+p01) * q11
            ({(0,0): 1, (0,1): 1}, {(1,1): 1},
             {(0,0): -1, (0,1): 1}),  # R00 -= S5, R01 += S5
            # S6: (p10-p00) * (q00+q01)
            ({(1,0): 1, (0,0): -1}, {(0,0): 1, (0,1): 1},
             {(1,1): 1}),  # R11 += S6
            # S7: (p01-p11) * (q10+q11)
            ({(0,1): 1, (1,1): -1}, {(1,0): 1, (1,1): 1},
             {(0,0): 1}),  # R00 += S7
        ]
        
        for s_idx, (a_coeffs, b_coeffs, c_coeffs) in enumerate(strassen_defs):
            col = col_offset + s_idx
            
            # For each element in the 2x2 A-side, determine which original matrix elements contribute
            for (ar, ac), a_coeff in a_coeffs.items():
                # ar, ac are positions in the 2x2 A-side matrix
                # The 2x2 A-side matrix is formed from a_blocks
                # a_blocks is a list of (block_row, block_col, sign)
                # The element at position (ar, ac) in the 2x2 A-side is:
                # sum over (br, bc, s) in a_blocks: s * A[2*br+ar, 2*bc+ac]
                for (br, bc, s) in a_blocks:
                    orig_row = 2 * br + ar
                    orig_col = 2 * bc + ac
                    U[orig_row * m + orig_col][col] += a_coeff * s
            
            # For each element in the 2x2 B-side, determine which original matrix elements contribute
            for (br, bc), b_coeff in b_coeffs.items():
                # br, bc are positions in the 2x2 B-side matrix
                for (bbr, bbc, s) in b_blocks:
                    orig_row = 2 * bbr + br
                    orig_col = 2 * bbc + bc
                    V[orig_row * p + orig_col][col] += b_coeff * s
            
            # For each element in the 2x2 C-side, determine which output elements receive the result
            for (cr, cc), c_coeff in c_coeffs.items():
                # cr, cc are positions in the 2x2 C-side matrix
                for (cbr, cbc, s) in c_blocks:
                    orig_row = 2 * cbr + cr
                    orig_col = 2 * cbc + cc
                    W[orig_col * n + orig_row][col] += c_coeff * s
    
    # Now define the 7 top-level Strassen products
    # Each has A-side blocks, B-side blocks, and C-side blocks
    
    # Block indices: 0=top-left, 1=top-right, 2=bottom-left, 3=bottom-right
    # Block (br, bc) corresponds to rows 2*br to 2*br+1, cols 2*bc to 2*bc+1
    
    # M1: A-side = A11 + A22, B-side = B11 + B22
    # A11 is block (0,0), A22 is block (1,1)
    # B11 is block (0,0), B22 is block (1,1)
    # C-side: C11 = M1 + ..., so M1 contributes to C11 (block (0,0))
    add_strassen_2x2(0,
                     a_blocks=[(0,0,1), (1,1,1)],  # A11 + A22
                     b_blocks=[(0,0,1), (1,1,1)],  # B11 + B22
                     c_blocks=[(0,0,1)])  # C11 += M1
    
    # M2: A-side = A21 + A22, B-side = B11
    # A21 is block (1,0), A22 is block (1,1)
    # B11 is block (0,0)
    # C-side: C21 = M2 + ..., C22 = ... - M2 + ...
    # So M2 contributes to C21 (block (1,0)) with +1 and C22 (block (1,1)) with -1
    add_strassen_2x2(7,
                     a_blocks=[(1,0,1), (1,1,1)],  # A21 + A22
                     b_blocks=[(0,0,1)],  # B11
                     c_blocks=[(1,0,1), (1,1,-1)])  # C21 += M2, C22 -= M2
    
    # M3: A-side = A11, B-side = B12 - B22
    # A11 is block (0,0)
    # B12 is block (0,1), B22 is block (1,1)
    # C-side: C12 = M3 + ..., C22 = ... + M3 + ...
    # So M3 contributes to C12 (block (0,1)) with +1 and C22 (block (1,1)) with +1
    add_strassen_2x2(14,
                     a_blocks=[(0,0,1)],  # A11
                     b_blocks=[(0,1,1), (1,1,-1)],  # B12 - B22
                     c_blocks=[(0,1,1), (1,1,1)])  # C12 += M3, C22 += M3
    
    # M4: A-side = A22, B-side = B21 - B11
    # A22 is block (1,1)
    # B21 is block (1,0), B11 is block (0,0)
    # C-side: C11 = ... + M4 + ..., C21 = ... + M4
    # So M4 contributes to C11 (block (0,0)) with +1 and C21 (block (1,0)) with +1
    add_strassen_2x2(21,
                     a_blocks=[(1,1,1)],  # A22
                     b_blocks=[(1,0,1), (0,0,-1)],  # B21 - B11
                     c_blocks=[(0,0,1), (1,0,1)])  # C11 += M4, C21 += M4
    
    # M5: A-side = A11 + A12, B-side = B22
    # A11 is block (0,0), A12 is block (0,1)
    # B22 is block (1,1)
    # C-side: C11 = ... - M5 + ..., C12 = ... + M5
    # So M5 contributes to C11 (block (0,0)) with -1 and C12 (block (0,1)) with +1
    add_strassen_2x2(28,
                     a_blocks=[(0,0,1), (0,1,1)],  # A11 + A12
                     b_blocks=[(1,1,1)],  # B22
                     c_blocks=[(0,0,-1), (0,1,1)])  # C11 -= M5, C12 += M5
    
    # M6: A-side = A21 - A11, B-side = B11 + B12
    # A21 is block (1,0), A11 is block (0,0)
    # B11 is block (0,0), B12 is block (0,1)
    # C-side: C22 = ... + M6
    # So M6 contributes to C22 (block (1,1)) with +1
    add_strassen_2x2(35,
                     a_blocks=[(1,0,1), (0,0,-1)],  # A21 - A11
                     b_blocks=[(0,0,1), (0,1,1)],  # B11 + B12
                     c_blocks=[(1,1,1)])  # C22 += M6
    
    # M7: A-side = A12 - A22, B-side = B21 + B22
    # A12 is block (0,1), A22 is block (1,1)
    # B21 is block (1,0), B22 is block (1,1)
    # C-side: C11 = ... + M7
    # So M7 contributes to C11 (block (0,0)) with +1
    add_strassen_2x2(42,
                     a_blocks=[(0,1,1), (1,1,-1)],  # A12 - A22
                     b_blocks=[(1,0,1), (1,1,1)],  # B21 + B22
                     c_blocks=[(0,0,1)])  # C11 += M7
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END

