"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 4, 4, 8
    # We use a block decomposition.
    # A is 4x4, split into 4 blocks of 2x2.
    # B is 4x8, split into 4 blocks of 2x4.
    # Each 2x4 block is split into 2 blocks of 2x2.
    # Total 8 Strassen 2x2x2 multiplications.
    # Rank = 8 * 7 = 56.
    
    rank = 56
    U = [[0] * rank for _ in range(n * m)] # 16 x 56
    V = [[0] * rank for _ in range(m * p)] # 32 x 56
    W = [[0] * rank for _ in range(p * n)] # 32 x 56
    
    # Strassen decomposition for 2x2x2
    # Indices for a 2x2 block:
    # A block: [[a, b], [c, d]] -> indices 0,1,2,3 in local 2x2
    # B block: [[e, f], [g, h]] -> indices 0,1,2,3 in local 2x2
    # C block: [[i, j], [k, l]] -> indices 0,1,2,3 in local 2x2
    # Strassen terms:
    # M1 = (a+d)(e+h)
    # M2 = (c+d)e
    # M3 = a(f-h)
    # M4 = d(g-f)
    # M5 = (a+b)h
    # M6 = (c-a)(e+f)
    # M7 = (b-d)(g+h)
    
    # We define the 7 Strassen products for a generic 2x2x2 block.
    # Each product is a rank-1 term in the local tensor.
    # We need to map these to the global indices.
    
    # Helper to add a Strassen decomposition for a specific 2x2x2 sub-problem
    # block_row: 0 or 1 (which 2x2 block of A in terms of rows)
    # block_col: 0 or 1 (which 2x2 block of A in terms of cols)
    # b_row: 0 or 1 (which 2x2 block of B in terms of rows)
    # b_col: 0 or 1 (which 2x2 block of B in terms of cols, 0 for left, 1 for right)
    # out_row: 0 or 1 (which 2x2 block of C in terms of rows)
    # out_col: 0 or 1 (which 2x2 block of C in terms of cols, 0 for left, 1 for right)
    # base_rank: starting rank index
    
    def add_strassen(block_row, block_col, b_row, b_col, out_row, out_col, base_rank):
        # Local indices for A block (2x2)
        # Global row i = block_row * 2 + local_i
        # Global col j = block_col * 2 + local_j
        
        # Local indices for B block (2x2)
        # Global row j = b_row * 2 + local_j
        # Global col k = b_col * 4 + local_k  (B is 4x8, so each 2x4 block is 2 rows x 4 cols)
        # Wait, B is split into 2x4 blocks. Each 2x4 block is split into two 2x2 blocks.
        # So b_col 0 means cols 0,1 of the 2x4 block. b_col 1 means cols 2,3.
        # Global col k = b_col * 4 + (b_sub_col * 2 + local_k)?
        # Let's clarify B indexing.
        # B is 4x8.
        # Block B_{br, bc} is rows [br*2, br*2+2) and cols [bc*4, bc*4+4).
        # Within B_{br, bc}, we split into left (cols 0,1) and right (cols 2,3).
        # So global col k = bc * 4 + b_sub_col * 2 + local_k.
        
        # Local indices for C block (2x2)
        # Global row i = out_row * 2 + local_i
        # Global col k = out_col * 4 + local_k  (C is 4x8)
        
        # Strassen coefficients for A (u), B (v), C (w)
        # Term 1: (a+d)(e+h)
        # u: a(0,0)=1, d(1,1)=1
        # v: e(0,0)=1, h(1,1)=1
        # w: i(0,0)=1, l(1,1)=1, j(0,1)=-1, k(1,0)=-1 ? No.
        # C = M1 - M2 - M3 + M4 ...
        # Let's use the standard Strassen reconstruction.
        # C00 = M1 - M4 + M5 - M7
        # C01 = M3 + M5
        # C10 = M2 + M4
        # C11 = M1 - M2 + M6 - M7
        
        # We define the 7 products and their contributions to C.
        
        products = [
            # (u_coeffs, v_coeffs, w_coeffs, w_sign)
            # u: [(local_i, local_j, val), ...]
            # v: [(local_j, local_k, val), ...]
            # w: [(local_i, local_k, val), ...]
            
            # M1 = (a+d)(e+h)
            {
                'u': [((0,0), 1), ((1,1), 1)],
                'v': [((0,0), 1), ((1,1), 1)],
                'w': [((0,0), 1), ((1,1), 1), ((0,1), -1), ((1,0), -1)] # This is wrong. Strassen doesn't work this way.
            }
        ]
        
        # Correct Strassen:
        # M1 = (A00+A11)(B00+B11)
        # M2 = (A10+A11)B00
        # M3 = A00(B01-B11)
        # M4 = A11(B10-B00)
        # M5 = (A00+A01)B11
        # M6 = (A10-A00)(B00+B01)
        # M7 = (A01-A11)(B10+B11)
        
        # C00 = M1 - M4 + M5 - M7
        # C01 = M3 + M5
        # C10 = M2 + M4
        # C11 = M1 - M2 + M6 - M7
        
        strassen_terms = [
            # M1
            {
                'u': [((0,0), 1), ((1,1), 1)],
                'v': [((0,0), 1), ((1,1), 1)],
                'w': [((0,0), 1), ((0,1), 0), ((1,0), 0), ((1,1), 1)] # Placeholder, need to handle signs in reconstruction
            },
            # M2
            {
                'u': [((1,0), 1), ((1,1), 1)],
                'v': [((0,0), 1)],
                'w': [((1,0), 1)]
            },
            # M3
            {
                'u': [((0,0), 1)],
                'v': [((0,1), 1), ((1,1), -1)],
                'w': [((0,1), 1)]
            },
            # M4
            {
                'u': [((1,1), 1)],
                'v': [((1,0), 1), ((0,0), -1)],
                'w': [((1,0), 1)]
            },
            # M5
            {
                'u': [((0,0), 1), ((0,1), 1)],
                'v': [((1,1), 1)],
                'w': [((0,0), 1), ((0,1), 1)]
            },
            # M6
            {
                'u': [((1,0), 1), ((0,0), -1)],
                'v': [((0,0), 1), ((0,1), 1)],
                'w': [((1,1), 1)]
            },
            # M7
            {
                'u': [((0,1), 1), ((1,1), -1)],
                'v': [((1,0), 1), ((1,1), 1)],
                'w': [((0,0), -1), ((1,1), -1)]
            }
        ]
        
        # This manual expansion is error-prone. 
        # I will use a known valid Strassen decomposition.
        
        # Standard Strassen:
        # P1 = (A00 + A11) * (B00 + B11)
        # P2 = (A10 + A11) * B00
        # P3 = A00 * (B01 - B11)
        # P4 = A11 * (B10 - B00)
        # P5 = (A00 + A01) * B11
        # P6 = (A10 - A00) * (B00 + B01)
        # P7 = (A01 - A11) * (B10 + B11)
        
        # C00 = P1 - P4 + P5 - P7
        # C01 = P3 + P5
        # C10 = P2 + P4
        # C11 = P1 - P2 + P6 - P7
        
        # We add 7 rank-1 terms for this 2x2x2 block.
        
        for t in range(7):
            r = base_rank + t
            
            # Define U, V, W contributions for this term
            
            # Term 1
            if t == 0:
                # U: A00 + A11
                U[block_row*4 + 0][r] += 1
                U[block_row*4 + 3][r] += 1
                # V: B00 + B11
                V[b_row*8 + 0][r] += 1
                V[b_row*8 + 3][r] += 1
                # W: C00 + C11 (with signs handled by other terms)
                # Actually, W should be the output of the product.
                # The product is P1.
                # P1 contributes to C00, C01, C10, C11.
                # It is easier to define the U and V vectors, and then determine the W vector.
                # The W vector is the outer product of the "result" of the linear forms?
                # No, the rank-1 tensor is u ⊗ v ⊗ w.
                # u is the linear form on A.
                # v is the linear form on B.
                # w is the linear form on C.
                
                # For P1 = (A00+A11)(B00+B11):
                # u = e00 + e11
                # v = e00 + e11
                # w = e00 + e11 ? No.
                # The product (A00+A11)(B00+B11) is a 2x2 matrix.
                # Let X = A00+A11, Y = B00+B11.
                # Z = X Y.
                # Z00 = X00 Y00 + X01 Y10
                # Z01 = X00 Y01 + X01 Y11
                # Z10 = X10 Y00 + X11 Y10
                # Z11 = X10 Y01 + X11 Y11
                # This is not a rank-1 tensor. It's a rank-4 tensor (schoolbook for 2x2).
                # Strassen replaces the 8 products with 7 "products" which are still matrix multiplications?
                # No, Strassen's "products" are scalar multiplications if the entries are scalars.
                # Here, the entries of A and B are scalars.
                # So P1 is a scalar product of two linear combinations of scalars.
                # P1 = (A00+A11) * (B00+B11).
                # This is a single scalar multiplication.
                # So the rank-1 tensor is:
                # u = e_A00 + e_A11
                # v = e_B00 + e_B11
                # w = e_C00 + e_C01 + e_C10 + e_C11 ? No.
                # The contribution of P1 to C is:
                # C00 += P1
                # C11 += P1
                # C01 -= P1 ? No.
                # Let's check the reconstruction formulas again.
                # C00 = P1 - P4 + P5 - P7
                # C01 = P3 + P5
                # C10 = P2 + P4
                # C11 = P1 - P2 + P6 - P7
                
                # So P1 contributes +1 to C00 and +1 to C11.
                # So w = e_C00 + e_C11.
                
                W[out_row*4 + 0][r] += 1
                W[out_row*4 + 3][r] += 1
                
            elif t == 1:
                # P2 = (A10+A11) * B00
                U[block_row*4 + 2][r] += 1
                U[block_row*4 + 3][r] += 1
                V[b_row*8 + 0][r] += 1
                # C10 += P2
                W[out_row*4 + 2][r] += 1
                
            elif t == 2:
                # P3 = A00 * (B01-B11)
                U[block_row*4 + 0][r] += 1
                V[b_row*8 + 1][r] += 1
                V[b_row*8 + 3][r] -= 1
                # C01 += P3
                W[out_row*4 + 1][r] += 1
                
            elif t == 3:
                # P4 = A11 * (B10-B00)
                U[block_row*4 + 3][r] += 1
                V[b_row*8 + 2][r] += 1
                V[b_row*8 + 0][r] -= 1
                # C10 += P4, C00 -= P4
                W[out_row*4 + 2][r] += 1
                W[out_row*4 + 0][r] -= 1
                
            elif t == 4:
                # P5 = (A00+A01) * B11
                U[block_row*4 + 0][r] += 1
                U[block_row*4 + 1][r] += 1
                V[b_row*8 + 3][r] += 1
                # C00 += P5, C01 += P5
                W[out_row*4 + 0][r] += 1
                W[out_row*4 + 1][r] += 1
                
            elif t == 5:
                # P6 = (A10-A00) * (B00+B01)
                U[block_row*4 + 2][r] += 1
                U[block_row*4 + 0][r] -= 1
                V[b_row*8 + 0][r] += 1
                V[b_row*8 + 1][r] += 1
                # C11 += P6
                W[out_row*4 + 3][r] += 1
                
            elif t == 6:
                # P7 = (A01-A11) * (B10+B11)
                U[block_row*4 + 1][r] += 1
                U[block_row*4 + 3][r] -= 1
                V[b_row*8 + 2][r] += 1
                V[b_row*8 + 3][r] += 1
                # C00 -= P7, C11 -= P7
                W[out_row*4 + 0][r] -= 1
                W[out_row*4 + 3][r] -= 1

    # We need to call add_strassen for each of the 8 sub-blocks.
    # The 4 output blocks are (0,0), (0,1), (1,0), (1,1).
    # Each output block is computed by 2 Strassen products (left and right halves of B).
    
    # Block (0,0): A blocks 0,0 and 0,1. B blocks 0,0 and 1,0.
    # Left half: B col 0. Right half: B col 1.
    
    # Output (0,0)
    # Left: A(0,0) * B(0,0) + A(0,1) * B(1,0) ... wait.
    # C00 = A00 B00 + A01 B10.
    # This is a sum of two 2x2 * 2x4 products?
    # No, A00 is 2x2, B00 is 2x4.
    # We split B00 into B00_L (2x2) and B00_R (2x2).
    # So we need Strassen for A00 * B00_L and A00 * B00_R.
    # And Strassen for A01 * B10_L and A01 * B10_R.
    
    # This means for each output block, we have 4 Strassen products?
    # C00 = A00 B00 + A01 B10.
    # A00 B00 is one 2x2 * 2x4 product. Rank 14.
    # A01 B10 is another 2x2 * 2x4 product. Rank 14.
    # Total rank for C00 is 28.
    # Total rank for 4 blocks is 112.
    
    # My previous analysis of rank 56 was incorrect.
    # The rank of 2x2 * 2x4 is 14.
    # C00 requires two such products.
    # So rank is 4 * 2 * 14 = 112.
    
    # This is not better than 128.
    # I will stick to the schoolbook 128.
    
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
