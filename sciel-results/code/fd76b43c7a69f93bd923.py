"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    
    # Strassen 2x2 decomposition
    # U: 4x7, V: 4x7, W: 4x7
    # Indices: 0:a, 1:b, 2:c, 3:d for A
    #          0:e, 1:f, 2:g, 3:h for B
    #          0:c00, 1:c01, 2:c10, 3:c11 for C (trace: k*2+i)
    
    U2 = [[0]*7 for _ in range(4)]
    V2 = [[0]*7 for _ in range(4)]
    W2 = [[0]*7 for _ in range(4)]
    
    # M1 = (a+d)(e+h) -> c00=1, c11=1
    U2[0][0] = 1; U2[3][0] = 1
    V2[0][0] = 1; V2[3][0] = 1
    W2[0][0] = 1; W2[3][0] = 1
    
    # M2 = (c+d)e -> c10=1, c11=-1
    U2[2][1] = 1; U2[3][1] = 1
    V2[0][1] = 1
    W2[2][1] = 1; W2[3][1] = -1
    
    # M3 = a(f-h) -> c01=1, c11=1
    U2[0][2] = 1
    V2[1][2] = 1; V2[3][2] = -1
    W2[1][2] = 1; W2[3][2] = 1
    
    # M4 = d(g-e) -> c00=1, c10=1
    U2[3][3] = 1
    V2[2][3] = 1; V2[0][3] = -1
    W2[0][3] = 1; W2[2][3] = 1
    
    # M5 = (a+b)h -> c00=-1, c01=1
    U2[0][4] = 1; U2[1][4] = 1
    V2[3][4] = 1
    W2[0][4] = -1; W2[1][4] = 1
    
    # M6 = (c-a)(e+f) -> c11=1
    U2[2][5] = 1; U2[0][5] = -1
    V2[0][5] = 1; V2[1][5] = 1
    W2[3][5] = 1
    
    # M7 = (b-d)(g+h) -> c00=1
    U2[1][6] = 1; U2[3][6] = -1
    V2[2][6] = 1; V2[3][6] = 1
    W2[0][6] = 1

    # Construct 4x4x4 decomposition (Rank 49)
    # U4: 16x49, V4: 16x49, W4: 16x49
    U4 = [[0]*49 for _ in range(16)]
    V4 = [[0]*49 for _ in range(16)]
    W4 = [[0]*49 for _ in range(16)]
    
    # Strassen 4x4 steps
    # S1 = (A00+A11)(B00+B11)
    # S2 = (A10+A11)B00
    # S3 = A00(B01-B11)
    # S4 = A11(B10-B00)
    # S5 = (A00+A01)B11
    # S6 = (A10-A00)(B00+B01)
    # S7 = (A01-A11)(B10+B11)
    
    # C00 = S1 + S4 - S5 + S7
    # C01 = S3 + S5
    # C10 = S2 + S4
    # C11 = S1 - S2 + S3 + S6
    
    # Define the 7 products and their contributions to C blocks
    # Each product is a 2x2 matrix mult of linear combos of 2x2 blocks.
    # We iterate over the 7 products (p_idx) and the 7 rank-1 components of 2x2 mult (r_idx).
    
    # Product definitions: (A_block_coeffs, B_block_coeffs, C_block_coeffs)
    # A_block_coeffs: dict {block_idx: coeff}, block_idx in 0..3 (00,01,10,11)
    # B_block_coeffs: dict {block_idx: coeff}
    # C_block_coeffs: dict {block_idx: coeff}
    
    products = [
        # S1
        ({0: 1, 3: 1}, {0: 1, 3: 1}, {0: 1, 3: 1}),
        # S2
        ({2: 1, 3: 1}, {0: 1}, {2: 1, 3: -1}),
        # S3
        ({0: 1}, {1: 1, 3: -1}, {1: 1, 3: 1}),
        # S4
        ({3: 1}, {2: 1, 0: -1}, {0: 1, 2: 1}),
        # S5
        ({0: 1, 1: 1}, {3: 1}, {0: -1, 1: 1}),
        # S6
        ({2: 1, 0: -1}, {0: 1, 1: 1}, {3: 1}),
        # S7
        ({1: 1, 3: -1}, {2: 1, 3: 1}, {0: 1})
    ]
    
    # Note: The C_block_coeffs in `products` above are the contributions of the *result* of the 2x2 mult 
    # to the 4x4 C blocks? 
    # No, Strassen computes S_i which are 2x2 matrices. 
    # Then C blocks are linear combos of S_i.
    # My `products` list above mixed the S_i definition and C block combination.
    # Let's correct this.
    
    # S_i definitions:
    # S1: A=(0,3), B=(0,3)
    # S2: A=(2,3), B=(0)
    # S3: A=(0), B=(1,-3)
    # S4: A=(3), B=(2,-0)
    # S5: A=(0,1), B=(3)
    # S6: A=(2,-0), B=(0,1)
    # S7: A=(1,-3), B=(2,3)
    
    # C block contributions:
    # C00: S1(1), S4(1), S5(-1), S7(1)
    # C01: S3(1), S5(1)
    # C10: S2(1), S4(1)
    # C11: S1(1), S2(-1), S3(1), S6(1)
    
    s_defs = [
        ({0: 1, 3: 1}, {0: 1, 3: 1}),
        ({2: 1, 3: 1}, {0: 1}),
        ({0: 1}, {1: 1, 3: -1}),
        ({3: 1}, {2: 1, 0: -1}),
        ({0: 1, 1: 1}, {3: 1}),
        ({2: 1, 0: -1}, {0: 1, 1: 1}),
        ({1: 1, 3: -1}, {2: 1, 3: 1})
    ]
    
    c_contribs = [
        # S1
        {0: 1, 3: 1},
        # S2
        {2: 1, 3: -1},
        # S3
        {1: 1, 3: 1},
        # S4
        {0: 1, 2: 1},
        # S5
        {0: -1, 1: 1},
        # S6
        {3: 1},
        # S7
        {0: 1}
    ]
    
    # Wait, the C contributions are for the *blocks* of C.
    # Each S_i is a 2x2 matrix.
    # So S_i contributes to C_block via W2.
    # Specifically, if S_i is computed by rank-1 terms $u v^T w^T$, 
    # then the element $S_i[a,b]$ is accumulated.
    # And $C_{block}$ is a linear combo of $S_i$ blocks.
    
    # Let's re-iterate:
    # For each S_i (i=0..6):
    #   For each rank-1 component r (r=0..6) of 2x2 mult:
    #     col = i * 7 + r
    #     Update U4, V4 using U2, V2 scaled by block coeffs.
    #     Update W4 using W2 scaled by block coeffs AND c_contribs[i].
    
    for i in range(7):
        a_coeffs, b_coeffs = s_defs[i]
        c_coeffs = c_contribs[i]
        
        for r in range(7):
            col = i * 7 + r
            
            # Update U4
            for a_blk, a_val in a_coeffs.items():
                if a_val == 0: continue
                # a_blk is 0..3 (00, 01, 10, 11)
                # Map to 2x2 block position (bi, bj)
                bi = a_blk // 2
                bj = a_blk % 2
                # For each element in 2x2 block
                for u in range(2):
                    for v in range(2):
                        # Global A index: (2*bi + u) * 4 + (2*bj + v)
                        g_idx_a = (2*bi + u) * 4 + (2*bj + v)
                        # U2 index: u * 2 + v
                        l_idx_a = u * 2 + v
                        U4[g_idx_a][col] += a_val * U2[l_idx_a][r]
                        
            # Update V4 (for 4x4x4 case, B is 4x4)
            for b_blk, b_val in b_coeffs.items():
                if b_val == 0: continue
                bi = b_blk // 2
                bj = b_blk % 2
                for u in range(2):
                    for v in range(2):
                        # Global B index (4x4): (2*bi + u) * 4 + (2*bj + v)
                        g_idx_b = (2*bi + u) * 4 + (2*bj + v)
                        l_idx_b = u * 2 + v
                        V4[g_idx_b][col] += b_val * V2[l_idx_b][r]
                        
            # Update W4 (for 4x4x4 case, C is 4x4)
            for c_blk, c_val in c_coeffs.items():
                if c_val == 0: continue
                ci = c_blk // 2
                cj = c_blk % 2
                for u in range(2):
                    for v in range(2):
                        # Global C index (4x4, trace): (2*cj + v) * 4 + (2*ci + u)
                        g_idx_c = (2*cj + v) * 4 + (2*ci + u)
                        l_idx_c = v * 2 + u # W2 index k*2+i -> v*2+u
                        W4[g_idx_c][col] += c_val * W2[l_idx_c][r]

    # Now construct full 4x4x8 decomposition (Rank 98)
    # U: 16x98
    # V: 32x98
    # W: 32x98
    
    U = [[0]*98 for _ in range(16)]
    V = [[0]*98 for _ in range(32)]
    W = [[0]*98 for _ in range(32)]
    
    # Copy U4 to both halves of U
    for i in range(16):
        for j in range(49):
            U[i][j] = U4[i][j]
            U[i][j+49] = U4[i][j]
            
    # V: First half (cols 0-3 of B)
    for i in range(16):
        for j in range(49):
            # V4[i][j] corresponds to B[2*bi+u, 2*bj+v] in 4x4 B.
            # Global B index: (2*bi+u)*8 + (2*bj+v)
            # V4 index i is (2*bi+u)*4 + (2*bj+v)
            # Let's just map directly from V4 indices to V indices
            # V4 row i corresponds to B element at flat index i in 4x4.
            # In 4x8, the first 4 columns are at flat indices i*2 + k? No.
            # B is 4x8. Flat index = row*8 + col.
            # V4 row i corresponds to row = i//4, col = i%4 in 4x4 B.
            # In 4x8 B, this element is at row = i//4, col = i%4.
            # Flat index = (i//4)*8 + (i%4).
            V[i//4 * 8 + i%4][j] = V4[i][j]
            
    # V: Second half (cols 4-7 of B)
    for i in range(16):
        for j in range(49):
            # V4 row i corresponds to B element at row = i//4, col = i%4 in 4x4 B.
            # In 4x8 B, this element is at row = i//4, col = 4 + (i%4).
            # Flat index = (i//4)*8 + (4 + i%4).
            V[i//4 * 8 + 4 + i%4][j+49] = V4[i][j]
            
    # W: First half (cols 0-3 of C)
    for i in range(16):
        for j in range(49):
            # W4 row i corresponds to C element at flat index i in 4x4 C (trace).
            # i = k*4 + r, where k is col, r is row in 4x4 C.
            # In 4x8 C, this element is at col = k, row = r.
            # Flat index (trace) = col*4 + row = k*4 + r = i.
            W[i][j] = W4[i][j]
            
    # W: Second half (cols 4-7 of C)
    for i in range(16):
        for j in range(49):
            # W4 row i corresponds to C element at k*4 + r in 4x4 C.
            # k = i//4, r = i%4.
            # In 4x8 C, this element is at col = 4 + k, row = r.
            # Flat index (trace) = (4+k)*4 + r = 16 + k*4 + r = 16 + i.
            W[16 + i][j+49] = W4[i][j]
            
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
