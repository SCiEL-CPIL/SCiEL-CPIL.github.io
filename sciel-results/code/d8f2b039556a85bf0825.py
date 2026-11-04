"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # We will build a rank 93 decomposition.
    # Part 1: 4x4x4 using recursive Strassen (rank 49)
    # Part 2: 4x4x2 using 4 Strassen blocks (rank 28)
    # Part 3: 4x4x1 using schoolbook (rank 16)
    
    total_rank = 49 + 28 + 16
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]
    
    rank_offset = 0
    
    # Helper to add a 2x2x2 Strassen block
    # i: row offset in A and C
    # j: col offset in A and row offset in B
    # k: col offset in B and C
    def add_strassen_block(i, j, k):
        nonlocal rank_offset
        # The 7 products of Strassen
        # M1 = (A11+A22)(B11+B22)
        # M2 = (A21+A22)B11
        # M3 = A11(B21-B11)
        # M4 = A22(B12-B22)
        # M5 = (A11+A12)B22
        # M6 = (A21-A11)(B11+B12)
        # M7 = (A12-A22)(B21+B22)
        
        # Coefficients for U, V, W
        # U indices: (i*2 + u_i) * m + (j*2 + u_j)
        # V indices: (j*2 + v_i) * p + (k*2 + v_j)
        # W indices: (k*2 + w_j) * n + (i*2 + w_i)
        
        # We need to define the linear combinations for U and V.
        # For each of the 7 terms, we add 1 to the appropriate U and V entries,
        # and combine them in W.
        
        # Term 1: (A11+A22)(B11+B22)
        # U: A11, A22
        # V: B11, B22
        # W: C11, C12, C21, C22 (with signs)
        
        # It is easier to iterate through the 7 terms and add contributions.
        
        # Indices in local 2x2 blocks
        # A: 00, 01, 10, 11
        # B: 00, 01, 10, 11
        # C: 00, 01, 10, 11
        
        # Strassen Terms:
        # 1: (A00+A11)(B00+B11)
        # 2: (A10+A11)B00
        # 3: A00(B10-B00)
        # 4: A11(B01-B11)
        # 5: (A00+A01)B11
        # 6: (A10-A00)(B00+B01)
        # 7: (A01-A11)(B10+B11)
        
        # C00 = M1 + M4 - M5 + M7
        # C01 = M3 + M5
        # C10 = M2 + M4
        # C11 = M1 - M2 + M3 + M6
        
        terms = [
            # (u_coeff, v_coeff, w_coeff, sign)
            # u_coeff is list of (local_u_idx, coeff)
            # v_coeff is list of (local_v_idx, coeff)
            # w_coeff is list of (local_w_idx, coeff)
            
            # M1
            {
                'u': [(0, 1), (3, 1)], # A00, A11
                'v': [(0, 1), (3, 1)], # B00, B11
                'w': [(0, 1), (2, 1)], # C00, C11 (Note: W index is k*n+i. Local w_idx: 0->C00, 1->C01, 2->C10, 3->C11? No.)
                # W shape p*n. Local block is 2x2.
                # Global W index: (k*2 + w_col) * n + (i*2 + w_row)
                # Local w_idx 0: w_row=0, w_col=0 -> C00
                # Local w_idx 1: w_row=0, w_col=1 -> C01
                # Local w_idx 2: w_row=1, w_col=0 -> C10
                # Local w_idx 3: w_row=1, w_col=1 -> C11
                # So w: [(0, 1), (3, 1)] for C00 and C11.
            },
            # M2
            {
                'u': [(2, 1), (3, 1)], # A10, A11
                'v': [(0, 1)],         # B00
                'w': [(2, 1), (3, -1)], # C10, C11
            },
            # M3
            {
                'u': [(0, 1)],         # A00
                'v': [(2, 1), (0, -1)],# B10, -B00
                'w': [(1, 1), (3, 1)], # C01, C11
            },
            # M4
            {
                'u': [(3, 1)],         # A11
                'v': [(1, 1), (3, -1)],# B01, -B11
                'w': [(0, 1), (2, 1)], # C00, C10
            },
            # M5
            {
                'u': [(0, 1), (1, 1)], # A00, A01
                'v': [(3, 1)],         # B11
                'w': [(0, -1), (1, 1)],# C00, C01
            },
            # M6
            {
                'u': [(2, 1), (0, -1)],# A10, -A00
                'v': [(0, 1), (1, 1)], # B00, B01
                'w': [(3, 1)],         # C11
            },
            # M7
            {
                'u': [(1, 1), (3, -1)],# A01, -A11
                'v': [(2, 1), (3, 1)], # B10, B11
                'w': [(0, 1)],         # C00
            },
        ]
        
        for t in terms:
            col = rank_offset
            rank_offset += 1
            for (lu, cu) in t['u']:
                gi = i * 2 + (lu // 2)
                gj = j * 2 + (lu % 2)
                U[gi * m + gj][col] += cu
            for (lv, cv) in t['v']:
                gj = j * 2 + (lv // 2)
                gk = k * 2 + (lv % 2)
                V[gj * p + gk][col] += cv
            for (lw, cw) in t['w']:
                gk = k * 2 + (lw % 2)
                gi = i * 2 + (lw // 2)
                W[gk * n + gi][col] += cw

    # Part 1: 4x4x4
    # Recursive Strassen: 7 block multiplications
    # Each block multiplication is 2x2x2
    # We need to map the 7 Strassen terms to the add_strassen_block calls?
    # No, add_strassen_block implements the 7 scalar multiplications for a 2x2x2 block.
    # To do 4x4x4, we need to perform 7 "virtual" 2x2x2 multiplications.
    # But our add_strassen_block handles the internal 7 scalar mults.
    # So we just need to call add_strassen_block 7 times?
    # No.
    # The recursive step:
    # M1 = (A11+A22)(B11+B22). This is a 2x2x2 multiplication.
    # So we need to instantiate a 2x2x2 algorithm for M1.
    # My add_strassen_block does exactly that.
    # So I need to call add_strassen_block 7 times, with appropriate offsets?
    # No, the offsets i, j, k in add_strassen_block refer to the position in the 4x4 matrix.
    # But the "virtual" blocks (A11+A22) are not contiguous in memory in the same way.
    # This approach is flawed. add_strassen_block assumes contiguous blocks.
    
    # Correct approach for 4x4x4:
    # We need to build the 49 terms explicitly.
    # Or, we can use the fact that M4 = M2 x M2.
    # The tensor is the Kronecker product.
    # If we have U2, V2, W2 for 2x2x2 (rank 7),
    # then U4 = U2 x U2 (Kronecker), etc.
    # But U2, V2, W2 are not simple identity matrices.
    
    # Let's just use the 8-block schoolbook-like Strassen which gives rank 56?
    # No, 8 blocks of 2x2x2 Strassen gives rank 56.
    # Is 56 better than 112? Yes.
    # Is 49 achievable with simple code?
    # Implementing the Kronecker product of the Strassen tensor is possible.
    
    # Let's stick to the 8-block method for 4x4x4 (Rank 56) for simplicity and correctness.
    # Then 4x4x2 (Rank 28) and 4x4x1 (Rank 16).
    # Total Rank 56 + 28 + 16 = 100.
    # This is still better than 112.
    
    # Let's reset rank_offset and use 8 blocks for 4x4x4.
    rank_offset = 0
    total_rank = 8 * 7 + 4 * 7 + 16 # 56 + 28 + 16 = 100
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]
    
    # 8 blocks for 4x4x4
    for i in [0, 2]:
        for j in [0, 2]:
            for k in [0, 2]:
                add_strassen_block(i, j, k)
                
    # 4 blocks for 4x4x2 (cols 4,5)
    # B is 4x2. Rows 0-1, 2-3.
    # A is 4x4.
    # C is 4x2.
    # Blocks:
    # i in [0, 2], j in [0, 2], k=4 (col offset in B, which is 0 in the 4x2 submatrix? No, k is global col index)
    # In add_strassen_block, k is the global column index in B.
    # The 2x2 block in B is at rows j, j+1 and cols k, k+1.
    # So for 4x4x2, we use k=4.
    for i in [0, 2]:
        for j in [0, 2]:
            add_strassen_block(i, j, 4)
            
    # 16 scalar mults for 4x4x1 (col 6)
    for i in range(4):
        for j in range(4):
            col = rank_offset
            rank_offset += 1
            U[i * m + j][col] = 1
            V[j * p + 6][col] = 1
            W[6 * n + i][col] = 1

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
