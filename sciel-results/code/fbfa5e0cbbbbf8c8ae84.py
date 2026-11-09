import time

DIMENSIONS = [4, 4, 7]

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin before 120s limit

    n, m, p = DIMENSIONS
    # We split p into p1=4 and p2=3
    p1 = 4
    p2 = 3
    
    U_parts = []
    V_parts = []
    W_parts = []

    # Part 1: 4x4x4 multiplication using Strassen (Rank 49)
    # We need to generate the rank-1 terms for 4x4x4
    # Recursive Strassen for 4x4:
    # C = AB, where A, B, C are 4x4.
    # Split into 2x2 blocks.
    # M1 = (A11 + A22)(B11 + B22)
    # M2 = (A21 + A22)B11
    # M3 = A11(B12 - B22)
    # M4 = A22(B21 - B11)
    # M5 = (A11 + A12)B22
    # M6 = (A21 - A11)(B11 + B12)
    # M7 = (A12 - A22)(B21 + B22)
    
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6

    # Each Mi is a 2x2x2 multiplication tensor.
    # We need to expand each 2x2x2 into 7 rank-1 terms (scalars).
    # A 2x2x2 multiplication C = AB has rank 7.
    # The 7 terms for 2x2x2 are defined by the Strassen algorithm for 2x2.
    
    def get_strassen_2x2_terms():
        # Returns list of (U_indices, V_indices, W_indices, coeff)
        # For 2x2x2:
        # U is 4x7, V is 4x7, W is 4x7
        # Indices for U: row = i*m + j, but here m=2, so i*2+j
        # Indices for V: row = j*p + k, but here p=2, so j*2+k
        # Indices for W: row = k*n + i, but here n=2, so k*2+i
        
        # Strassen 2x2:
        # M1 = (A11+A22)(B11+B22)
        # M2 = (A21+A22)B11
        # M3 = A11(B12-B22)
        # M4 = A22(B21-B11)
        # M5 = (A11+A12)B22
        # M6 = (A21-A11)(B11+B12)
        # M7 = (A12-A22)(B21+B22)
        
        # C11 = M1 + M4 - M5 + M7
        # C12 = M3 + M5
        # C21 = M2 + M4
        # C22 = M1 - M2 + M3 + M6
        
        # We need to express each M as a sum of rank-1 terms? No, M IS the product of two 2x2 "matrices" which are linear combinations of A and B.
        # Wait, the rank of 2x2x2 is 7. This means we can decompose the tensor into 7 rank-1 tensors.
        # The standard Strassen decomposition provides this.
        # Each "product" in Strassen is a scalar multiplication of two linear forms? No, they are 2x2 matrix multiplications?
        # No, for 2x2x2, the "inputs" are scalars if we view it recursively? No.
        # The rank of 2x2x2 is 7. This is achieved by 7 terms.
        # Let's list the 7 rank-1 terms for the 2x2x2 tensor.
        # The tensor $T_{2,2,2}$ has components $T_{ijk}$ where $C_{ij} = \sum_k A_{ik} B_{kj}$.
        # The 7 terms are usually given by the Strassen algorithm applied to the 2x2 blocks?
        # Actually, the rank 7 of 2x2x2 is exactly the Strassen algorithm itself if we consider the inputs as 1x1 matrices? No.
        # The Strassen algorithm computes 2x2 matrix multiplication using 7 scalar multiplications.
        # So, the 7 rank-1 terms correspond to these 7 scalar multiplications.
        
        # Let's define the 7 terms for 2x2x2.
        # Term t=0..6
        # For each term, we need coefficients for A (4 entries), B (4 entries), C (4 entries).
        
        # A = [[a0, a1], [a2, a3]]
        # B = [[b0, b1], [b2, b3]]
        # C = [[c0, c1], [c2, c3]]
        
        # Strassen products:
        # p0 = (a0+a3)(b0+b3)
        # p1 = (a2+a3)b0
        # p2 = a0(b1-b3)
        # p3 = a3(b2-b0)
        # p4 = (a0+a1)b3
        # p5 = (a2-a0)(b0+b1)
        # p6 = (a1-a3)(b2+b3)
        
        # C entries:
        # c0 = p0 + p3 - p4 + p6
        # c1 = p2 + p4
        # c2 = p1 + p3
        # c3 = p0 - p1 + p2 + p5
        
        # We construct U, V, W for this 2x2x2 sub-problem.
        # U rows: 4 (a0..a3). V rows: 4 (b0..b3). W rows: 4 (c0..c3).
        # There are 7 columns (p0..p6).
        
        U_222 = [[0]*7 for _ in range(4)]
        V_222 = [[0]*7 for _ in range(4)]
        W_222 = [[0]*7 for _ in range(4)]
        
        # p0
        U_222[0][0] = 1; U_222[3][0] = 1
        V_222[0][0] = 1; V_222[3][0] = 1
        W_222[0][0] = 1; W_222[3][0] = 1
        
        # p1
        U_222[2][1] = 1; U_222[3][1] = 1
        V_222[0][1] = 1
        W_222[2][1] = 1; W_222[3][1] = 1
        
        # p2
        U_222[0][2] = 1
        V_222[1][2] = 1; V_222[3][2] = -1
        W_222[1][2] = 1; W_222[3][2] = 1
        
        # p3
        U_222[3][3] = 1
        V_222[2][3] = 1; V_222[0][3] = -1
        W_222[0][3] = 1; W_222[2][3] = 1
        
        # p4
        U_222[0][4] = 1; U_222[1][4] = 1
        V_222[3][4] = 1
        W_222[1][4] = 1; W_222[3][4] = 1
        
        # p5
        U_222[2][5] = 1; U_222[0][5] = -1
        V_222[0][5] = 1; V_222[1][5] = 1
        W_222[3][5] = 1
        
        # p6
        U_222[1][6] = 1; U_222[3][6] = -1
        V_222[2][6] = 1; V_222[3][6] = 1
        W_222[0][6] = 1; W_222[3][6] = 1
        
        return U_222, V_222, W_222

    U_222, V_222, W_222 = get_strassen_2x2_terms()
    
    # Now, construct 4x4x4 using 4x 2x2x2 blocks?
    # No, the Strassen algorithm for 4x4 uses 7 "products" of 2x2 matrices.
    # Each 2x2 matrix multiplication is a 2x2x2 tensor.
    # So we have 7 such tensors.
    # The total rank is 7 * 7 = 49.
    
    # We need to map the global 4x4 indices to the local 2x2 indices for each of the 7 blocks.
    
    # Global A is 4x4. Indices A[i][j] with i,j in 0..3.
    # Global B is 4x4. Indices B[j][k] with j,k in 0..3.
    # Global C is 4x4. Indices C[i][k] with i,k in 0..3.
    
    # Strassen 4x4:
    # Blocks of A: A11 (0,0), A12 (0,1), A21 (1,0), A22 (1,1) -> (row_block, col_block)
    # A11 = A[0:2, 0:2]
    # A12 = A[0:2, 2:4]
    # A21 = A[2:4, 0:2]
    # A22 = A[2:4, 2:4]
    
    # Similarly for B and C.
    
    # The 7 products M1..M7 are 2x2 matrix multiplications.
    # M1 = (A11 + A22) * (B11 + B22)
    # M2 = (A21 + A22) * B11
    # M3 = A11 * (B12 - B22)
    # M4 = A22 * (B21 - B11)
    # M5 = (A11 + A12) * B22
    # M6 = (A21 - A11) * (B11 + B12)
    # M7 = (A12 - A22) * (B21 + B22)
    
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    # Each Mk is a 2x2x2 multiplication.
    # We have the U_222, V_222, W_222 for a generic 2x2x2 multiplication.
    # For each Mk, we instantiate this template, scaling/offsetting the indices.
    
    # Helper to add a block to U, V, W
    # U_global is 16xRank. V_global is 16xRank. W_global is 16xRank.
    # Current rank for part 1 will be 49.
    
    rank_part1 = 49
    U1 = [[0]*rank_part1 for _ in range(16)]
    V1 = [[0]*rank_part1 for _ in range(16)]
    W1 = [[0]*rank_part1 for _ in range(16)]
    
    # For each of the 7 Strassen products, we have 7 scalar multiplications (from 2x2x2 rank 7).
    # Total 49 columns.
    
    # Mapping of Strassen product index (0..6) to local 2x2 indices
    # Product t uses U_222, V_222, W_222.
    # The "A" part of product t corresponds to a linear combination of A blocks.
    # The "B" part corresponds to a linear combination of B blocks.
    # The "C" part contributes to C blocks.
    
    # We define the coefficients for A, B, C for each Strassen product t.
    # A_t = alpha * A_r1c1 + ...
    # B_t = beta * B_r1c1 + ...
    # C_t contributes to C_r1c1 ...
    
    # Strassen Products Definition:
    # t=0: M1. A: A11+A22. B: B11+B22. C: C11(+1), C22(+1)
    # t=1: M2. A: A21+A22. B: B11. C: C21(+1), C22(-1)
    # t=2: M3. A: A11. B: B12-B22. C: C12(+1), C22(+1)
    # t=3: M4. A: A22. B: B21-B11. C: C11(+1), C21(+1)
    # t=4: M5. A: A11+A12. B: B22. C: C11(-1), C12(+1)
    # t=5: M6. A: A21-A11. B: B11+B12. C: C22(+1)
    # t=6: M7. A: A12-A22. B: B21+B22. C: C11(+1)
    
    # Note: C contributions are additive.
    
    # A blocks indices (row_block, col_block):
    # A11: (0,0), A12: (0,1), A21: (1,0), A22: (1,1)
    
    strassen_defs = [
        # (A_blocks, B_blocks, C_contribs)
        # A_blocks: list of (rb, cb, coeff)
        # B_blocks: list of (rb, cb, coeff)
        # C_contribs: list of (rb, cb, coeff)
        (
            [((0,0), 1), ((1,1), 1)], # A11 + A22
            [((0,0), 1), ((1,1), 1)], # B11 + B22
            [((0,0), 1), ((1,1), 1)]  # C11, C22
        ),
        (
            [((1,0), 1), ((1,1), 1)], # A21 + A22
            [((0,0), 1)],             # B11
            [((1,0), 1), ((1,1), -1)]# C21, C22
        ),
        (
            [((0,0), 1)],             # A11
            [((0,1), 1), ((1,1), -1)],# B12 - B22
            [((0,1), 1), ((1,1), 1)] # C12, C22
        ),
        (
            [((1,1), 1)],             # A22
            [((1,0), 1), ((0,0), -1)],# B21 - B11
            [((0,0), 1), ((1,0), 1)] # C11, C21
        ),
        (
            [((0,0), 1), ((0,1), 1)], # A11 + A12
            [((1,1), 1)],             # B22
            [((0,0), -1), ((0,1), 1)]# C11, C12
        ),
        (
            [((1,0), 1), ((0,0), -1)],# A21 - A11
            [((0,0), 1), ((0,1), 1)],# B11 + B12
            [((1,1), 1)]             # C22
        ),
        (
            [((0,1), 1), ((1,1), -1)],# A12 - A22
            [((1,0), 1), ((1,1), 1)],# B21 + B22
            [((0,0), 1)]             # C11
        )
    ]
    
    # Fill U1, V1, W1
    for t in range(7):
        A_blocks, B_blocks, C_contribs = strassen_defs[t]
        # For each of the 7 rank-1 terms in the 2x2x2 core (u_idx, v_idx, w_idx in 0..6)
        for core_idx in range(7):
            col = t * 7 + core_idx
            
            # Get coefficients from core 2x2x2 decomposition
            # U_222[row][core_idx] gives coeff for A element row
            # V_222[row][core_idx] gives coeff for B element row
            # W_222[row][core_idx] gives coeff for C element row
            
            # Map core 2x2 indices to global 4x4 block indices
            # Core A index r (0..3) corresponds to local row r//2, col r%2 within the 2x2 block.
            # We need to sum contributions to global U from all A_blocks.
            
            # Initialize local contribution to global U for this column
            # U_global[row][col] += sum( coeff_A_block * U_222[local_row][core_idx] )
            
            # A part
            for (arb, acb), coeff_A in A_blocks:
                # Global row index for A: arb*2 + r, acb*2 + c?
                # A is 4x4. Row index = r*4+c.
                # Block (arb, acb) covers rows arb*2 .. arb*2+1, cols acb*2 .. acb*2+1.
                # Local index r_l (0..3) in 2x2x2 corresponds to local row r_l//2, col r_l%2.
                # Global row = (arb*2 + r_l//2) * 4 + (acb*2 + r_l%2)
                
                for r_l in range(4):
                    val = U_222[r_l][core_idx]
                    if val != 0:
                        gr = arb * 2 + r_l // 2
                        gc = acb * 2 + r_l % 2
                        global_row = gr * 4 + gc
                        U1[global_row][col] += coeff_A * val
            
            # B part
            # V is 4x4. Row index = j*4+k.
            # Block (brb, bcb) covers rows brb*2..brb*2+1, cols bcb*2..bcb*2+1.
            # Local index c_l (0..3) in 2x2x2 corresponds to local row c_l//2, col c_l%2.
            # Global row = (brb*2 + c_l//2) * 4 + (bcb*2 + c_l%2)
            
            for (brb, bcb), coeff_B in B_blocks:
                for c_l in range(4):
                    val = V_222[c_l][core_idx]
                    if val != 0:
                        gr = brb * 2 + c_l // 2
                        gc = bcb * 2 + c_l % 2
                        global_row = gr * 4 + gc
                        V1[global_row][col] += coeff_B * val
            
            # C part
            # W is 4x4. Row index = k*4+i (Trace convention W[k*n+i]).
            # Block (crb, ccb) covers rows (output row i) crb*2..crb*2+1, cols (output col k) ccb*2..ccb*2+1?
            # Wait, W index is k*n+i.
            # C is 4x4. C[i][k].
            # Block (crb, ccb) covers i in crb*2..crb*2+1, k in ccb*2..ccb*2+1.
            # Local index w_l (0..3) in 2x2x2 corresponds to local row i_l = w_l//2, local col k_l = w_l%2.
            # Global i = crb*2 + i_l, Global k = ccb*2 + k_l.
            # Global W row index = k*4 + i = (ccb*2 + k_l)*4 + (crb*2 + i_l).
            
            for (crb, ccb), coeff_C in C_contribs:
                for w_l in range(4):
                    val = W_222[w_l][core_idx]
                    if val != 0:
                        gi = crb * 2 + w_l // 2
                        gk = ccb * 2 + w_l % 2
                        global_row = gk * 4 + gi
                        W1[global_row][col] += coeff_C * val
                        
    # Part 2: 4x4x3 multiplication using Schoolbook (Rank 48)
    # p2 = 3.
    # U2 is 16x48, V2 is 16x48 (actually 4*3=12 rows? No, V is m*p. m=4, p=3. So 12 rows).
    # Wait, V dimension is m*p. In the split, we treat the 3 columns of B as a 4x3 matrix.
    # So V2 has 4*3=12 rows.
    # W2 has p*n = 3*4=12 rows.
    # U2 has n*m = 16 rows.
    
    rank_part2 = 48
    U2 = [[0]*rank_part2 for _ in range(16)]
    V2 = [[0]*rank_part2 for _ in range(12)]
    W2 = [[0]*rank_part2 for _ in range(12)]
    
    col2 = 0
    for i in range(4):
        for j in range(4):
            for k in range(3):
                U2[i*4+j][col2] = 1
                V2[j*3+k][col2] = 1
                W2[k*4+i][col2] = 1
                col2 += 1

    # Combine parts
    # Total rank = 49 + 48 = 97
    total_rank = rank_part1 + rank_part2
    
    U = [[0]*total_rank for _ in range(16)]
    V = [[0]*total_rank for _ in range(28)] # 4*7
    W = [[0]*total_rank for _ in range(28)] # 7*4
    
    # Copy Part 1 (cols 0-48)
    for r in range(16):
        for c in range(49):
            U[r][c] = U1[r][c]
    for r in range(16):
        for c in range(49):
            V[r][c] = V1[r][c]
    for r in range(16):
        for c in range(49):
            W[r][c] = W1[r][c]
            
    # Copy Part 2 (cols 49-96)
    # U rows 0-15
    for r in range(16):
        for c in range(48):
            U[r][49+c] = U2[r][c]
    # V rows 0-11 (from Part 2) go to V rows 16-27?
    # Part 1 V was 4x4. Rows 0-15.
    # Part 2 V is 4x3. Rows 0-11.
    # These correspond to B columns 4,5,6.
    # In the full B (4x7), these are rows 16..27?
    # B index = j*7 + k.
    # Part 1 used k=0..3. So rows j*7+0 .. j*7+3.
    # Part 2 uses k=4..6. So rows j*7+4 .. j*7+6.
    # V2 row index j*3+k_l. k_l is 0..2.
    # Global V row = j*7 + (4+k_l).
    
    for r2 in range(12):
        j = r2 // 3
        k_l = r2 % 3
        global_row = j * 7 + 4 + k_l
        for c in range(48):
            V[global_row][49+c] = V2[r2][c]
            
    # W Part 2
    # W is 7x4. Rows k*4+i.
    # Part 1 used k=0..3. Rows 0..15.
    # Part 2 uses k=4..6. Rows 16..27.
    # W2 row index k_l*4+i. k_l is 0..2.
    # Global W row = (4+k_l)*4 + i.
    
    for r2 in range(12):
        k_l = r2 // 4
        i = r2 % 4
        global_row = (4 + k_l) * 4 + i
        for c in range(48):
            W[global_row][49+c] = W2[r2][c]

    return {"U": U, "V": V, "W": W}

