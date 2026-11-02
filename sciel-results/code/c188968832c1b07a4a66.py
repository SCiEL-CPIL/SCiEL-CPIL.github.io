"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # We use Strassen's algorithm for 4x4x4, which has rank 49.
    # For 4x4x8, we split B into two 4x4 blocks, so rank is 49 + 49 = 98.
    
    # First, define the 2x2x2 Strassen decomposition (rank 7)
    # U2: 4x7, V2: 4x7, W2: 4x7
    # Indices for 2x2: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
    U2 = [
        [1, 0, 1, 0, 1, -1, 0], # a
        [0, 0, 0, 0, 1, 0, 1],  # b
        [0, 1, 0, 0, 0, 1, 0],  # c
        [1, 1, 0, 1, 0, 0, -1]  # d
    ]
    V2 = [
        [1, 1, 0, 0, 0, 1, 0], # e
        [0, 0, 1, 0, 0, 1, 0], # f
        [0, 0, 0, 1, 0, 0, 1], # g
        [1, 0, -1, -1, 1, 0, 1] # h
    ]
    W2 = [
        [1, 0, 0, 1, -1, 0, 1], # C00
        [0, 0, 1, 0, 1, 0, 0],  # C01
        [0, 1, 0, 1, 0, 0, 0],  # C10
        [1, -1, 1, 0, 0, 1, 0]  # C11
    ]
    
    # Verify W2:
    # C00 = S1 + S4 - S5 + S7 -> [1, 0, 0, 1, -1, 0, 1]
    # C01 = S3 + S5 -> [0, 0, 1, 0, 1, 0, 0]
    # C10 = S2 + S4 -> [0, 1, 0, 1, 0, 0, 0]
    # C11 = S1 - S2 + S3 + S6 -> [1, -1, 1, 0, 0, 1, 0]
    # Matches.

    # Now construct 4x4x4 decomposition (rank 49)
    # Strassen products M1..M7 for 4x4x4
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B12-B22)
    # M4 = A22(B21-B11)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    
    # For each Mi, we need to map the 2x2 indices to 4x4 indices.
    # A is 4x4. Blocks:
    # A11: rows 0-1, cols 0-1 -> indices 0,1,4,5
    # A12: rows 0-1, cols 2-3 -> indices 2,3,6,7
    # A21: rows 2-3, cols 0-1 -> indices 8,9,12,13
    # A22: rows 2-3, cols 2-3 -> indices 10,11,14,15
    
    # B is 4x4. Blocks:
    # B11: rows 0-1, cols 0-1 -> indices 0,1,4,5
    # B12: rows 0-1, cols 2-3 -> indices 2,3,6,7
    # B21: rows 2-3, cols 0-1 -> indices 8,9,12,13
    # B22: rows 2-3, cols 2-3 -> indices 10,11,14,15
    
    # C is 4x4. Blocks:
    # C11: rows 0-1, cols 0-1 -> indices 0,1,4,5
    # C12: rows 0-1, cols 2-3 -> indices 2,3,6,7
    # C21: rows 2-3, cols 0-1 -> indices 8,9,12,13
    # C22: rows 2-3, cols 2-3 -> indices 10,11,14,15

    rank_444 = 49
    U444 = [[0] * rank_444 for _ in range(16)]
    V444 = [[0] * rank_444 for _ in range(16)]
    W444 = [[0] * rank_444 for _ in range(16)]
    
    # Define block index mappings
    # 2x2 index (r, c) in block -> 4x4 index
    # Block A11: (0,0)->0, (0,1)->1, (1,0)->4, (1,1)->5
    # Block A12: (0,0)->2, (0,1)->3, (1,0)->6, (1,1)->7
    # Block A21: (0,0)->8, (0,1)->9, (1,0)->12, (1,1)->13
    # Block A22: (0,0)->10, (0,1)->11, (1,0)->14, (1,1)->15
    
    def map_2x2_to_4x4(block, r, c):
        # block: 0=A11, 1=A12, 2=A21, 3=A22
        # r, c: 0 or 1
        base = block * 4 + r * 2 + c
        # Wait, 4x4 index is i*4+j.
        # A11: i=0,1; j=0,1. idx = i*4+j.
        # A12: i=0,1; j=2,3. idx = i*4+j.
        # A21: i=2,3; j=0,1. idx = i*4+j.
        # A22: i=2,3; j=2,3. idx = i*4+j.
        
        if block == 0: # A11
            return r * 4 + c
        elif block == 1: # A12
            return r * 4 + (c + 2)
        elif block == 2: # A21
            return (r + 2) * 4 + c
        elif block == 3: # A22
            return (r + 2) * 4 + (c + 2)
            
    # Strassen coefficients for A and B for each Mi
    # Mi = (sum coeff_A[k] * A_block[k]) * (sum coeff_B[k] * B_block[k])
    # We need to expand this into linear combinations of 2x2 entries.
    
    # For each Mi, we have 7 sub-products S1..S7.
    # Each Sj uses U2[j], V2[j], W2[j].
    # U2[j] is a vector of 4 entries for the 2x2 matrix A^(i).
    # A^(i) is a linear combination of 4x4 blocks.
    # So for each entry in U2[j], we add it to the corresponding 4x4 indices in U444.
    
    strassen_A_coeffs = [
        # M1: A11 + A22
        [[1, 0, 0, 1]], 
        # M2: A21 + A22
        [[0, 0, 1, 1]],
        # M3: A11
        [[1, 0, 0, 0]],
        # M4: A22
        [[0, 0, 0, 1]],
        # M5: A11 + A12
        [[1, 1, 0, 0]],
        # M6: A21 - A11
        [[-1, 0, 1, 0]],
        # M7: A12 - A22
        [[0, 1, 0, -1]]
    ]
    
    strassen_B_coeffs = [
        # M1: B11 + B22
        [[1, 0, 0, 1]],
        # M2: B11
        [[1, 0, 0, 0]],
        # M3: B12 - B22
        [[0, 1, 0, -1]],
        # M4: B21 - B11
        [[-1, 0, 1, 0]],
        # M5: B22
        [[0, 0, 0, 1]],
        # M6: B11 + B12
        [[1, 1, 0, 0]],
        # M7: B21 + B22
        [[0, 0, 1, 1]]
    ]
    
    # C coefficients
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    # For each Mi, we add to W444.
    # W2[j] is for the 2x2 output of Mi.
    # We need to add W2[j] to the corresponding block of C.
    # Which block of C does Mi contribute to?
    # M1 contributes to C11 and C22.
    # M2 contributes to C21 and C22.
    # M3 contributes to C12 and C22.
    # M4 contributes to C11 and C21.
    # M5 contributes to C11 and C12.
    # M6 contributes to C22.
    # M7 contributes to C11.
    
    # Let's define for each Mi, which C blocks it affects and with what sign.
    # And for each C block, which Mi affects it.
    
    # It's easier to iterate over Mi, and for each sub-product Sj,
    # determine the contribution to C.
    
    # C11 gets: M1 (S1,S4,-S5,S7), M4 (S1,S4), -M5 (S1,-S5), M7 (S7)
    # This is getting complicated. Let's just use the W2 matrix.
    # W2[j] gives the contribution of Sj to the 2x2 output of Mi.
    # So for Mi, the 2x2 output C^(i) is computed.
    # Then C^(i) is added to the appropriate blocks of the global C.
    
    # For M1: C^(1) is added to C11 and C22.
    # C11 += C^(1)
    # C22 += C^(1)
    
    # For M2: C^(2) is added to C21 and C22.
    # C21 += C^(2)
    # C22 -= C^(2)
    
    # For M3: C^(3) is added to C12 and C22.
    # C12 += C^(3)
    # C22 += C^(3)
    
    # For M4: C^(4) is added to C11 and C21.
    # C11 += C^(4)
    # C21 += C^(4)
    
    # For M5: C^(5) is added to C11 and C12.
    # C11 -= C^(5)
    # C12 += C^(5)
    
    # For M6: C^(6) is added to C22.
    # C22 += C^(6)
    
    # For M7: C^(7) is added to C11.
    # C11 += C^(7)
    
    # So for each Mi, we know which C blocks it maps to and with what sign.
    # Let's define a list of (block_index, sign) for each Mi.
    # block_index: 0=C11, 1=C12, 2=C21, 3=C22
    
    c_blocks_for_M = [
        [(0, 1), (3, 1)],       # M1 -> C11, C22
        [(2, 1), (3, -1)],      # M2 -> C21, C22
        [(1, 1), (3, 1)],       # M3 -> C12, C22
        [(0, 1), (2, 1)],       # M4 -> C11, C21
        [(0, -1), (1, 1)],      # M5 -> C11, C12
        [(3, 1)],               # M6 -> C22
        [(0, 1)]                # M7 -> C11
    ]
    
    for i in range(7):
        # i is 0..6 corresponding to M1..M7
        A_coeffs = strassen_A_coeffs[i][0]
        B_coeffs = strassen_B_coeffs[i][0]
        
        for j in range(7):
            # j is 0..6 corresponding to S1..S7
            col = i * 7 + j
            
            # U444
            for r in range(2):
                for c in range(2):
                    val = U2[r][j]
                    if val != 0:
                        for b_idx in range(4):
                            coeff = A_coeffs[b_idx]
                            if coeff != 0:
                                idx = map_2x2_to_4x4(b_idx, r, c)
                                U444[idx][col] += val * coeff
            
            # V444
            for r in range(2):
                for c in range(2):
                    val = V2[r][j]
                    if val != 0:
                        for b_idx in range(4):
                            coeff = B_coeffs[b_idx]
                            if coeff != 0:
                                idx = map_2x2_to_4x4(b_idx, r, c)
                                V444[idx][col] += val * coeff
            
            # W444
            for r in range(2):
                for c in range(2):
                    val = W2[r][j]
                    if val != 0:
                        for block_idx, sign in c_blocks_for_M[i]:
                            idx = map_2x2_to_4x4(block_idx, r, c)
                            W444[idx][col] += val * sign

    # Now we have U444, V444, W444 for 4x4x4.
    # We need 4x4x8.
    # Split B into B_L (cols 0-3) and B_R (cols 4-7).
    # C_L = A B_L, C_R = A B_R.
    # C_L is 4x4, C_R is 4x4.
    
    rank_total = 98
    U = [[0] * rank_total for _ in range(16)]
    V = [[0] * rank_total for _ in range(32)]
    W = [[0] * rank_total for _ in range(32)]
    
    # Copy U444 to first 49 cols of U
    for i in range(16):
        for j in range(49):
            U[i][j] = U444[i][j]
            U[i][j + 49] = U444[i][j]
            
    # Copy V444 to V
    # V_L corresponds to B_L (rows 0-15 of V)
    # V_R corresponds to B_R (rows 16-31 of V)
    for i in range(16):
        for j in range(49):
            V[i][j] = V444[i][j]
            V[i + 16][j + 49] = V444[i][j]
            
    # Copy W444 to W
    # W_L corresponds to C_L (rows 0-15 of W)
    # W_R corresponds to C_R (rows 16-31 of W)
    for i in range(16):
        for j in range(49):
            W[i][j] = W444[i][j]
            W[i + 16][j + 49] = W444[i][j]
            
    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
