"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 9]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    
    # Rank 49 decomposition for 4x4x4 using Strassen
    # We construct U4, V4, W4 of shape (16, 49), (16, 49), (16, 49)
    
    # M2 Strassen decomposition (rank 7)
    # Indices for 2x2: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
    # U2, V2, W2 are 4x7
    
    U2 = [[0]*7 for _ in range(4)]
    V2 = [[0]*7 for _ in range(4)]
    W2 = [[0]*7 for _ in range(4)]
    
    # t=0: M1
    U2[0][0] = 1; U2[3][0] = 1
    V2[0][0] = 1; V2[3][0] = 1
    W2[0][0] = 1; W2[3][0] = 1
    
    # t=1: M2
    U2[2][1] = 1; U2[3][1] = 1
    V2[0][1] = 1
    W2[2][1] = 1; W2[3][1] = -1
    
    # t=2: M3
    U2[0][2] = 1
    V2[1][2] = 1; V2[3][2] = -1
    W2[1][2] = 1; W2[3][2] = 1
    
    # t=3: M4
    U2[3][3] = 1
    V2[2][3] = 1; V2[0][3] = -1
    W2[0][3] = 1; W2[2][3] = 1
    
    # t=4: M5
    U2[0][4] = 1; U2[1][4] = 1
    V2[3][4] = 1
    W2[0][4] = -1; W2[1][4] = 1
    
    # t=5: M6
    U2[2][5] = 1; U2[0][5] = -1
    V2[1][5] = 1; V2[2][5] = 1
    W2[3][5] = 1
    
    # t=6: M7
    U2[1][6] = 1; U2[2][6] = -1
    V2[0][6] = 1; V2[2][6] = 1
    W2[0][6] = 1
    
    # Build M4 (4x4x4) from M2
    # A blocks: A00 (0,1,4,5), A01 (2,3,6,7), A10 (8,9,12,13), A11 (10,11,14,15)
    # B blocks: B00 (0,1,4,5), B01 (2,3,6,7), B10 (8,9,12,13), B11 (10,11,14,15)
    # C blocks: C00 (0,1,4,5), C01 (2,3,6,7), C10 (8,9,12,13), C11 (10,11,14,15)
    
    # Map 2x2 index to 4x4 index
    # block (r, c) of 2x2 corresponds to 4x4 indices:
    # r=0, c=0: 0,1,4,5
    # r=0, c=1: 2,3,6,7
    # r=1, c=0: 8,9,12,13
    # r=1, c=1: 10,11,14,15
    
    def get_4x4_indices(block_r, block_c):
        # block_r, block_c in {0,1}
        # returns list of 4 indices in 4x4 matrix
        base = block_r * 8 + block_c * 2
        return [base, base+1, base+4, base+5]
        
    # Actually, 4x4 flattened:
    # 0 1 2 3
    # 4 5 6 7
    # 8 9 10 11
    # 12 13 14 15
    
    # Block (0,0): 0,1,4,5
    # Block (0,1): 2,3,6,7
    # Block (1,0): 8,9,12,13
    # Block (1,1): 10,11,14,15
    
    def block_indices(br, bc):
        if br == 0 and bc == 0: return [0,1,4,5]
        if br == 0 and bc == 1: return [2,3,6,7]
        if br == 1 and bc == 0: return [8,9,12,13]
        if br == 1 and bc == 1: return [10,11,14,15]
        
    # Strassen M4 terms S1..S7
    # S1 = (A00+A11)(B00+B11)
    # S2 = (A10+A11)B00
    # S3 = A00(B01-B11)
    # S4 = A11(B10-B00)
    # S5 = (A00+A01)B11
    # S6 = (A10-A00)(B01+B10)
    # S7 = (A01-A10)(B00+B10)
    
    # C00 = S1 + S4 - S5 + S7
    # C01 = S3 + S5
    # C10 = S2 + S4
    # C11 = S1 - S2 + S3 + S6
    
    # For each Sk, we use M2 decomposition.
    # U4, V4, W4 will be 16x49.
    
    U4 = [[0]*49 for _ in range(16)]
    V4 = [[0]*49 for _ in range(16)]
    W4 = [[0]*49 for _ in range(16)]
    
    # Define the "effective" 2x2 matrices for each S
    # Each S is a product of linear combinations of blocks.
    # We need to map the 2x2 indices of U2/V2/W2 to the 4x4 indices.
    
    # For a given S, we define:
    # U_eff: 4x7 (from U2) mapped to 4x4 indices
    # V_eff: 4x7 (from V2) mapped to 4x4 indices
    # W_eff: 4x7 (from W2) mapped to 4x4 indices
    
    # We'll create a helper to add a scaled M2 decomposition to U4/V4/W4
    # for specific block combinations.
    
    def add_m2_term(s_idx, u_blocks, v_blocks, w_blocks, scale=1):
        # s_idx: 0..6 for the 7 terms of M2
        # u_blocks: list of (block_r, block_c, coeff) for A
        # v_blocks: list of (block_r, block_c, coeff) for B
        # w_blocks: list of (block_r, block_c, coeff) for C
        
        col_offset = s_idx * 7
        # Wait, S_k uses the full M2 decomposition.
        # So for each S_k, we add 7 columns to U4/V4/W4.
        # Total 7 * 7 = 49 columns.
        
        # For each of the 7 M2 terms (t=0..6):
        for t in range(7):
            col = col_offset + t
            
            # U contribution
            for (br, bc, coeff) in u_blocks:
                idxs = block_indices(br, bc)
                # Map 2x2 index in U2[t] to 4x4 index
                # U2 row index is 0..3 corresponding to 2x2 (0,0), (0,1), (1,0), (1,1)
                # We need to add U2[row][t] * coeff to U4[4x4_index][col]
                
                # U2[row][t] is the coefficient for the 2x2 element
                # The 2x2 elements are mapped to 4x4 blocks.
                # U2[0] is (0,0) of the 2x2 matrix.
                # This corresponds to the (0,0) element of the block.
                # The block has 4 elements.
                # U2[0] -> element (0,0) of block -> 4x4 index idxs[0]
                # U2[1] -> element (0,1) of block -> 4x4 index idxs[1]
                # U2[2] -> element (1,0) of block -> 4x4 index idxs[2]
                # U2[3] -> element (1,1) of block -> 4x4 index idxs[3]
                
                for local_idx in range(4):
                    u_val = U2[local_idx][t]
                    if u_val != 0:
                        U4[idxs[local_idx]][col] += scale * u_val * coeff
                        
            # V contribution
            for (br, bc, coeff) in v_blocks:
                idxs = block_indices(br, bc)
                for local_idx in range(4):
                    v_val = V2[local_idx][t]
                    if v_val != 0:
                        V4[idxs[local_idx]][col] += scale * v_val * coeff
                        
            # W contribution
            for (br, bc, coeff) in w_blocks:
                idxs = block_indices(br, bc)
                for local_idx in range(4):
                    w_val = W2[local_idx][t]
                    if w_val != 0:
                        W4[idxs[local_idx]][col] += scale * w_val * coeff

    # S1 = (A00+A11)(B00+B11)
    add_m2_term(0, [(0,0,1), (1,1,1)], [(0,0,1), (1,1,1)], [(0,0,1), (1,1,1)], 1)
    
    # S2 = (A10+A11)B00
    add_m2_term(1, [(1,0,1), (1,1,1)], [(0,0,1)], [(1,0,1), (1,1,1)], 1)
    
    # S3 = A00(B01-B11)
    add_m2_term(2, [(0,0,1)], [(0,1,1), (1,1,-1)], [(0,1,1), (1,1,1)], 1)
    
    # S4 = A11(B10-B00)
    add_m2_term(3, [(1,1,1)], [(1,0,1), (0,0,-1)], [(0,0,1), (1,0,1)], 1)
    
    # S5 = (A00+A01)B11
    add_m2_term(4, [(0,0,1), (0,1,1)], [(1,1,1)], [(0,0,-1), (0,1,1)], 1)
    
    # S6 = (A10-A00)(B01+B10)
    add_m2_term(5, [(1,0,1), (0,0,-1)], [(0,1,1), (1,0,1)], [(1,1,1)], 1)
    
    # S7 = (A01-A10)(B00+B10)
    add_m2_term(6, [(0,1,1), (1,0,-1)], [(0,0,1), (1,0,1)], [(0,0,1)], 1)
    
    # Now we have U4, V4, W4 for one 4x4x4 multiplication.
    
    # Construct full U, V, W for 4x4x9
    # p=9. Split into p1=4, p2=4, p3=1.
    # Rank = 49 + 49 + 16 = 114.
    
    rank = 114
    U = [[0]*rank for _ in range(16)]
    V = [[0]*rank for _ in range(36)]
    W = [[0]*rank for _ in range(36)]
    
    # Part 1: Columns 0-3 of B
    # V indices: j*p + k. p=9.
    # j in 0..3, k in 0..3.
    # V4 has rows 0..15 (j*4+k).
    # We need to map V4 rows to V rows.
    # V4 row r corresponds to j = r//4, k = r%4.
    # V row index = j*9 + k.
    
    for r in range(16):
        j = r // 4
        k = r % 4
        v_row = j * 9 + k
        for col in range(49):
            U[r][col] = U4[r][col]
            V[v_row][col] = V4[r][col]
            W[r][col] = W4[r][col] # W4 rows are C indices (i*4+k). i=r//4, k=r%4. W row i*9+k?
            # Wait, W4 is for C (4x4). W4 row index is i*4+k.
            # W full row index is i*9+k.
            # So W[r][col] should be W4[r][col] mapped to W[i*9+k][col].
            
    # Correct W mapping for Part 1
    for r in range(16):
        i = r // 4
        k = r % 4
        w_row = i * 9 + k
        for col in range(49):
            W[w_row][col] = W4[r][col]
            
    # Part 2: Columns 4-7 of B
    # V indices: j in 0..3, k in 4..7.
    # V4 row r (j, k_local) -> V row j*9 + (k_local+4).
    # W indices: i in 0..3, k in 4..7.
    # W4 row r (i, k_local) -> W row i*9 + (k_local+4).
    
    for r in range(16):
        j = r // 4
        k = r % 4
        v_row = j * 9 + (k + 4)
        for col in range(49, 98):
            U[r][col] = U4[r][col - 49]
            V[v_row][col] = V4[r][col - 49]
            
    for r in range(16):
        i = r // 4
        k = r % 4
        w_row = i * 9 + (k + 4)
        for col in range(49, 98):
            W[w_row][col] = W4[r][col - 49]
            
    # Part 3: Column 8 of B
    # Standard 4x4x1 decomposition. Rank 16.
    # U: 16x16. U[i*4+j, i*4+j] = 1.
    # V: 4x16. V[j, r] = 1 if r%4 == j.
    # W: 4x16. W[i, r] = 1 if r//4 == i.
    
    for r in range(16):
        i = r // 4
        j = r % 4
        U[r][98 + r] = 1
        # V row j, col 98+r
        V[j * 9 + 8][98 + r] = 1
        # W row i, col 98+r
        W[i * 9 + 8][98 + r] = 1
        
    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
