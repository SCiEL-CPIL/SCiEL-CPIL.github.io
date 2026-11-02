"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    
    # We will use a block decomposition approach.
    # n=4, m=4, p=5.
    # We can view A as 2x2 blocks of 2x2.
    # We can view B as 2x2 blocks of 2x2, plus an extra column (since p=5 is odd).
    # Specifically:
    # A is partitioned into 4 blocks: A11, A12, A21, A22 (each 2x2)
    # B is partitioned into 3 column-groups: B_left (2x2), B_mid (2x2), B_right (2x1).
    # Wait, let's align dimensions carefully.
    # A: 4x4. B: 4x5. C: 4x5.
    
    # Let's split the matrix multiplication C = A B into blocks based on A's rows and B's columns.
    # Row blocks of A/C:
    # R0: rows 0,1
    # R1: rows 2,3
    # Col blocks of A/B:
    # C0: cols 0,1
    # C1: cols 2,3
    # Col blocks of B (extra):
    # C2: col 4
    
    # So A is:
    # [A00 A01]
    # [A10 A11]
    # where Aij are 2x2.
    
    # B is:
    # [B00 B01 | b0]  <-- 2x2, 2x2, 2x1
    # [B10 B11 | b1]
    # where B00, B01, B10, B11 are 2x2, and b0, b1 are 2x1 (vectors).
    
    # C is:
    # [C00 C01 | c0]
    # [C10 C11 | c1]
    
    # Equations:
    # C00 = A00 B00 + A01 B10
    # C01 = A00 B01 + A01 B11
    # c0  = A00 b0  + A01 b1
    # C10 = A10 B00 + A11 B10
    # C11 = A10 B01 + A11 B11
    # c1  = A10 b0  + A11 b1
    
    # We can apply Strassen's algorithm to the 2x2 matrix multiplications.
    # Strassen works for multiplying two 2x2 matrices using 7 scalar multiplications.
    # Here, the "scalars" are actually vectors or small matrices, but Strassen's linear combinations work on the elements.
    
    # Let's use Strassen for the 2x2 x 2x2 products.
    # For the 2x2 x 2x1 products, we can just do schoolbook (4 mults each) or see if there's a better way.
    # Actually, 2x2 by 2x1 is just a linear combination of two column vectors.
    # Rank of 2x2 by 2x1 tensor is 4 (it's just dot products).
    
    # Let's implement Strassen for the 2x2 parts.
    # Let X and Y be 2x2 matrices.
    # S1 = (X00 + X11) (Y00 + Y11)
    # S2 = (X10 + X11) Y00
    # S3 = X00 (Y01 - Y11)
    # S4 = X11 (Y10 - Y00)
    # S5 = (X00 + X01) Y11
    # S6 = (X10 - X00) (Y00 + Y01)
    # S7 = (X01 - X11) (Y10 + Y11)
    #
    # X Y = [ S1 + S4 - S5 + S7,  S3 + S5 ]
    #        [ S2 + S4,            S1 - S2 + S3 + S6 ]
    
    # We need to combine these for our specific blocks.
    # C00 = A00 B00 + A01 B10. This is a sum of two matrix products.
    # We can compute M1 = A00 B00 using Strassen (rank 7).
    # M2 = A01 B10 using Strassen (rank 7).
    # Then C00 = M1 + M2.
    # Since the decomposition is linear, we can just sum the rank-7 decompositions.
    # Total rank for C00 will be 7 + 7 = 14? No, the multiplications are independent.
    # Yes, because the terms in the tensor sum are orthogonal in the factor space if we allocate separate columns for each term.
    
    # Let's list all the products required:
    # 1. A00 * B00 (2x2 * 2x2) -> Strassen (7 mults)
    # 2. A01 * B10 (2x2 * 2x2) -> Strassen (7 mults)
    # 3. A00 * B01 (2x2 * 2x2) -> Strassen (7 mults)
    # 4. A01 * B11 (2x2 * 2x2) -> Strassen (7 mults)
    # 5. A10 * B00 (2x2 * 2x2) -> Strassen (7 mults)
    # 6. A11 * B10 (2x2 * 2x2) -> Strassen (7 mults)
    # 7. A10 * B01 (2x2 * 2x2) -> Strassen (7 mults)
    # 8. A11 * B11 (2x2 * 2x2) -> Strassen (7 mults)
    #
    # Plus the vector products (2x2 * 2x1):
    # 9.  A00 * b0
    # 10. A01 * b1
    # 11. A10 * b0
    # 12. A11 * b1
    # Each is rank 4. (4 * 4 = 16 mults)
    
    # Total rank = 8 * 7 + 16 = 56 + 16 = 72.
    # Schoolbook was 80. 72 < 80. This is an improvement.
    
    # We need to construct U, V, W matrices.
    # Dimensions:
    # U: 16 x 72
    # V: 20 x 72
    # W: 20 x 72
    
    # Helper to add a Strassen block.
    # We need to map global indices to local indices for the 2x2 blocks.
    
    # Global Indexing:
    # A: [i, j] where i in 0..3, j in 0..3.
    # U index: i * m + j = 4 * i + j.
    #
    # B: [k, l] where k in 0..3, l in 0..4.
    # V index: k * p + l = 5 * k + l.
    #
    # C: [r, c] where r in 0..3, c in 0..4.
    # W index: c * n + r = 5 * c + r.
    
    # Strassen Block Constructor
    # We need to define the coefficients for U, V, W for each of the 7 multiplications in Strassen.
    # For a generic 2x2 product X*Y = Z:
    # Z00 = S1 + S4 - S5 + S7
    # Z01 = S3 + S5
    # Z10 = S2 + S4
    # Z11 = S1 - S2 + S3 + S6
    
    # S1 = (X00 + X11)(Y00 + Y11)
    # S2 = (X10 + X11)Y00
    # S3 = X00(Y01 - Y11)
    # S4 = X11(Y10 - Y00)
    # S5 = (X00 + X01)Y11
    # S6 = (X10 - X00)(Y00 + Y01)
    # S7 = (X01 - X11)(Y10 + Y11)
    
    # We will create a list of "terms". Each term corresponds to one multiplication in the tensor decomposition.
    # A term is defined by:
    # - u_coeffs: list of (global_u_idx, coeff)
    # - v_coeffs: list of (global_v_idx, coeff)
    # - w_coeffs: list of (global_w_idx, coeff)
    
    terms = []
    
    # Function to get global U index for A block element
    # A blocks:
    # A00: A[0..1, 0..1]
    # A01: A[0..1, 2..3]
    # A10: A[2..3, 0..1]
    # A11: A[2..3, 2..3]
    
    def get_u_indices(block_row, block_col, local_r, local_c):
        # block_row: 0 or 1
        # block_col: 0 or 1
        # local_r: 0 or 1
        # local_c: 0 or 1
        i = block_row * 2 + local_r
        j = block_col * 2 + local_c
        return i * m + j
    
    # Function to get global V index for B block element
    # B blocks (left part, 2x2):
    # B00: B[0..1, 0..1]
    # B01: B[0..1, 2..3]
    # B10: B[2..3, 0..1]
    # B11: B[2..3, 2..3]
    # Note: B has 5 cols. We treat cols 0,1 and 2,3 as blocks.
    
    def get_v_indices_block(block_row, block_col, local_r, local_c):
        # block_row: 0 or 1 (rows of B)
        # block_col: 0 or 1 (col group of B: 0 for cols 0,1; 1 for cols 2,3)
        k = block_row * 2 + local_r
        l = block_col * 2 + local_c
        return k * p + l
    
    # Function to get global W index for C block element
    # C blocks (left part, 2x2):
    # C00: C[0..1, 0..1]
    # C01: C[0..1, 2..3]
    # C10: C[2..3, 0..1]
    # C11: C[2..3, 2..3]
    
    def get_w_indices_block(block_row, block_col, local_r, local_c):
        # block_row: 0 or 1 (rows of C)
        # block_col: 0 or 1 (col group of C: 0 for cols 0,1; 1 for cols 2,3)
        r = block_row * 2 + local_r
        c = block_col * 2 + local_c
        return c * n + r
    
    # Add Strassen terms for a 2x2 matrix product
    # X_block_r, X_block_c: row/col block of A
    # Y_block_r, Y_block_c: row/col block of B (0 for cols 0-1, 1 for cols 2-3)
    # Z_block_r, Z_block_c: row/col block of C
    def add_strassen(a_br, a_bc, b_br, b_bc, c_br, c_bc):
        # Define the 7 Strassen multiplications
        # Each is a tuple: (u_coeffs, v_coeffs, w_coeffs)
        
        # S1 = (X00 + X11)(Y00 + Y11)
        # X00 is local (0,0) of A_block. Global: get_u_indices(a_br, a_bc, 0, 0)
        # X11 is local (1,1) of A_block. Global: get_u_indices(a_br, a_bc, 1, 1)
        # Y00 is local (0,0) of B_block. Global: get_v_indices_block(b_br, b_bc, 0, 0)
        # Y11 is local (1,1) of B_block. Global: get_v_indices_block(b_br, b_bc, 1, 1)
        
        u1 = [(get_u_indices(a_br, a_bc, 0, 0), 1), (get_u_indices(a_br, a_bc, 1, 1), 1)]
        v1 = [(get_v_indices_block(b_br, b_bc, 0, 0), 1), (get_v_indices_block(b_br, b_bc, 1, 1), 1)]
        # W coeffs are determined by how S1 contributes to Z
        # Z00 += S1, Z11 += S1
        w1 = [(get_w_indices_block(c_br, c_bc, 0, 0), 1), (get_w_indices_block(c_br, c_bc, 1, 1), 1)]
        terms.append((u1, v1, w1))
        
        # S2 = (X10 + X11)Y00
        u2 = [(get_u_indices(a_br, a_bc, 1, 0), 1), (get_u_indices(a_br, a_bc, 1, 1), 1)]
        v2 = [(get_v_indices_block(b_br, b_bc, 0, 0), 1)]
        # Z10 += S2
        w2 = [(get_w_indices_block(c_br, c_bc, 1, 0), 1)]
        terms.append((u2, v2, w2))
        
        # S3 = X00(Y01 - Y11)
        u3 = [(get_u_indices(a_br, a_bc, 0, 0), 1)]
        v3 = [(get_v_indices_block(b_br, b_bc, 0, 1), 1), (get_v_indices_block(b_br, b_bc, 1, 1), -1)]
        # Z01 += S3, Z11 += S3
        w3 = [(get_w_indices_block(c_br, c_bc, 0, 1), 1), (get_w_indices_block(c_br, c_bc, 1, 1), 1)]
        terms.append((u3, v3, w3))
        
        # S4 = X11(Y10 - Y00)
        u4 = [(get_u_indices(a_br, a_bc, 1, 1), 1)]
        v4 = [(get_v_indices_block(b_br, b_bc, 1, 0), 1), (get_v_indices_block(b_br, b_bc, 0, 0), -1)]
        # Z00 += S4, Z10 += S4
        w4 = [(get_w_indices_block(c_br, c_bc, 0, 0), 1), (get_w_indices_block(c_br, c_bc, 1, 0), 1)]
        terms.append((u4, v4, w4))
        
        # S5 = (X00 + X01)Y11
        u5 = [(get_u_indices(a_br, a_bc, 0, 0), 1), (get_u_indices(a_br, a_bc, 0, 1), 1)]
        v5 = [(get_v_indices_block(b_br, b_bc, 1, 1), 1)]
        # Z00 -= S5, Z01 += S5
        w5 = [(get_w_indices_block(c_br, c_bc, 0, 0), -1), (get_w_indices_block(c_br, c_bc, 0, 1), 1)]
        terms.append((u5, v5, w5))
        
        # S6 = (X10 - X00)(Y00 + Y01)
        u6 = [(get_u_indices(a_br, a_bc, 1, 0), 1), (get_u_indices(a_br, a_bc, 0, 0), -1)]
        v6 = [(get_v_indices_block(b_br, b_bc, 0, 0), 1), (get_v_indices_block(b_br, b_bc, 0, 1), 1)]
        # Z11 += S6
        w6 = [(get_w_indices_block(c_br, c_bc, 1, 1), 1)]
        terms.append((u6, v6, w6))
        
        # S7 = (X01 - X11)(Y10 + Y11)
        u7 = [(get_u_indices(a_br, a_bc, 0, 1), 1), (get_u_indices(a_br, a_bc, 1, 1), -1)]
        v7 = [(get_v_indices_block(b_br, b_bc, 1, 0), 1), (get_v_indices_block(b_br, b_bc, 1, 1), 1)]
        # Z00 += S7
        w7 = [(get_w_indices_block(c_br, c_bc, 0, 0), 1)]
        terms.append((u7, v7, w7))
        
    # Add vector multiplication terms (2x2 A * 2x1 B = 2x1 C)
    # A_block is 2x2, B_block is 2x1 (col 4 of B, rows 0,1 or 2,3).
    # C_block is 2x1 (col 4 of C, rows 0,1 or 2,3).
    def add_vec_mul(a_br, b_br, c_br):
        # A_block: rows 2*a_br, 2*a_br+1.
        # B_block: rows 2*b_br, 2*b_br+1. Col 4.
        # C_block: rows 2*c_br, 2*c_br+1. Col 4.
        
        # C_r = sum_k A_rk B_k
        # C_0 = A_00 B_0 + A_01 B_1
        # C_1 = A_10 B_0 + A_11 B_1
        
        # Term 1: A_00 B_0
        u = [(get_u_indices(a_br, 0, 0, 0), 1)]
        v = [(2*b_br*5 + 4, 1)] # B row 2*b_br, col 4
        w = [(4*5 + (2*c_br), 1)] # C col 4, row 2*c_br
        terms.append((u, v, w))
        
        # Term 2: A_01 B_1
        u = [(get_u_indices(a_br, 0, 0, 1), 1)]
        v = [( (2*b_br+1)*5 + 4, 1)] # B row 2*b_br+1, col 4
        w = [(4*5 + (2*c_br), 1)] # C col 4, row 2*c_br
        terms.append((u, v, w))
        
        # Term 3: A_10 B_0
        u = [(get_u_indices(a_br, 0, 1, 0), 1)]
        v = [(2*b_br*5 + 4, 1)]
        w = [(4*5 + (2*c_br+1), 1)] # C col 4, row 2*c_br+1
        terms.append((u, v, w))
        
        # Term 4: A_11 B_1
        u = [(get_u_indices(a_br, 0, 1, 1), 1)]
        v = [( (2*b_br+1)*5 + 4, 1)]
        w = [(4*5 + (2*c_br+1), 1)]
        terms.append((u, v, w))
        
    # Now call the constructors for all blocks.
    # C00 = A00 B00 + A01 B10
    add_strassen(0, 0, 0, 0, 0, 0) # A00 * B00 -> C00
    add_strassen(0, 1, 1, 0, 0, 0) # A01 * B10 -> C00
    
    # C01 = A00 B01 + A01 B11
    add_strassen(0, 0, 0, 1, 0, 1) # A00 * B01 -> C01
    add_strassen(0, 1, 1, 1, 0, 1) # A01 * B11 -> C01
    
    # C10 = A10 B00 + A11 B10
    add_strassen(1, 0, 0, 0, 1, 0) # A10 * B00 -> C10
    add_strassen(1, 1, 1, 0, 1, 0) # A11 * B10 -> C10
    
    # C11 = A10 B01 + A11 B11
    add_strassen(1, 0, 0, 1, 1, 1) # A10 * B01 -> C11
    add_strassen(1, 1, 1, 1, 1, 1) # A11 * B11 -> C11
    
    # Vector parts (Col 4)
    # c0 = A00 b0 + A01 b1
    # Note: b0 is B rows 0,1 col 4. b1 is B rows 2,3 col 4.
    # Our add_vec_mul(a_br, b_br, c_br) handles one 2x2 * 2x1 product.
    # We have TWO products for each output block: A_top * b_bottom AND A_top * b_top?
    # Wait.
    # C_top (rows 0,1) = A_top * B_top + A_bottom * B_bottom?
    # No.
    # A is 4x4. B is 4x5.
    # Rows of A are 0..3. Rows of B are 0..3.
    # A is partitioned into row blocks A0 (rows 0,1) and A1 (rows 2,3).
    # B is partitioned into row blocks B0 (rows 0,1) and B1 (rows 2,3).
    # C rows 0,1 correspond to linear combinations of A rows 0,1.
    # C_rows_0_1 = A_rows_0_1 * B_cols_4? NO.
    # C_{i, 4} = sum_{k=0}^3 A_{i, k} B_{k, 4}.
    # For i=0: C_{0,4} = A_{0,0}B_{0,4} + A_{0,1}B_{1,4} + A_{0,2}B_{2,4} + A_{0,3}B_{3,4}.
    # A_{0,0}, A_{0,1} are in A00. A_{0,2}, A_{0,3} are in A01.
    # So C_{0,4} = (A00_col0 * b0_row0 + A00_col0 * b0_row1) ... wait.
    # Let's use the block vectors.
    # b0 = [B_{0,4}, B_{1,4}]^T.
    # b1 = [B_{2,4}, B_{3,4}]^T.
    # A00 is 2x2. A00 * b0 is a vector of length 2.
    # A01 is 2x2. A01 * b1 is a vector of length 2.
    # The sum is the top half of C's 4th column.
    # So yes, we need A00*b0 + A01*b1.
    
    add_vec_mul(0, 0, 0) # A00 * b0 -> c0
    add_vec_mul(0, 1, 0) # A01 * b1 -> c0
    
    # For c1 (rows 2,3):
    add_vec_mul(1, 0, 1) # A10 * b0 -> c1
    add_vec_mul(1, 1, 1) # A11 * b1 -> c1
    
    rank = len(terms)
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for col, (u_terms, v_terms, w_terms) in enumerate(terms):
        for idx, coeff in u_terms:
            U[idx][col] += coeff
        for idx, coeff in v_terms:
            V[idx][col] += coeff
        for idx, coeff in w_terms:
            W[idx][col] += coeff
            
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
