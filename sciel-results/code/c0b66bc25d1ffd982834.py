import time

DIMENSIONS = [4, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 100  # 100s safety margin
    
    n, m, p = DIMENSIONS
    
    # We will build the decomposition by summing 4 sub-tensors:
    # 1. 4x4x4 (Rank 49, Strassen)
    # 2. 4x4x1 (Rank 16, Schoolbook)
    # 3. 4x1x4 (Rank 16, Schoolbook)
    # 4. 4x1x1 (Rank 4, Schoolbook)
    
    # Helper to create schoolbook decomposition for n x m x p
    def schoolbook(n, m, p):
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return U, V, W

    # Helper to create Strassen decomposition for 2x2x2
    def strassen_2x2():
        # Returns U, V, W as lists of 7 vectors of length 4
        # Each vector corresponds to a linear combination of the 4 elements
        # A = [[A00, A01], [A10, A11]]
        # B = [[B00, B01], [B10, B11]]
        # C = [[C00, C01], [C10, C11]]
        
        # M1 = (A00+A11)(B00+B11)
        # M2 = (A10+A11)B00
        # M3 = A00(B01-B11)
        # M4 = A11(B10-B00)
        # M5 = (A00+A01)B11
        # M6 = (A10-A00)(B00+B01)
        # M7 = (A01-A11)(B10+B11)
        
        # U vectors (A coeffs)
        # Index: 0:00, 1:01, 2:10, 3:11
        U = [
            [1, 0, 0, 1], # M1
            [0, 0, 1, 1], # M2
            [1, 0, 0, 0], # M3
            [0, 0, 0, 1], # M4
            [1, 1, 0, 0], # M5
            [-1, 0, 1, 0], # M6
            [0, 1, 0, -1]  # M7
        ]
        
        # V vectors (B coeffs)
        V = [
            [1, 0, 0, 1], # M1
            [1, 0, 0, 0], # M2
            [0, 1, 0, -1], # M3
            [0, 0, 1, -1], # M4
            [0, 0, 0, 1], # M5
            [1, 1, 0, 0], # M6
            [0, 0, 1, 1]  # M7
        ]
        
        # W vectors (C coeffs)
        # C00 = M1 + M4 - M5 + M7
        # C01 = M3 + M5
        # C10 = M2 + M4
        # C11 = M1 - M2 + M3 + M6
        
        W = [
            [1, 0, 0, 1], # M1 contributes to C00, C11
            [0, 0, 1, -1], # M2 contributes to C10, C11
            [0, 1, 0, 1], # M3 contributes to C01, C11
            [1, 0, 1, 0], # M4 contributes to C00, C10
            [-1, 1, 0, 0], # M5 contributes to C00, C01
            [0, 0, 0, 1], # M6 contributes to C11
            [1, 0, 0, 0]  # M7 contributes to C00
        ]
        
        # Convert to matrices of shape (4, 7)
        U_mat = [[0]*7 for _ in range(4)]
        V_mat = [[0]*7 for _ in range(4)]
        W_mat = [[0]*7 for _ in range(4)]
        
        for r in range(7):
            for i in range(4):
                U_mat[i][r] = U[r][i]
                V_mat[i][r] = V[r][i]
                W_mat[i][r] = W[r][i]
                
        return U_mat, V_mat, W_mat

    # Helper to create Strassen decomposition for 4x4x4
    def strassen_4x4():
        # Recursively use strassen_2x2
        # The 4x4 matrix is split into 2x2 blocks.
        # A = [[A11, A12], [A21, A22]] where Aij are 2x2.
        # The 7 products P1..P7 are 2x2 matrix multiplications.
        # Each Pk is decomposed into 7 terms using strassen_2x2.
        
        # We need to map the local 2x2 indices to global 4x4 indices.
        # Global A index: i*4 + j.
        # Block A11: i in 0,1; j in 0,1.
        # Block A12: i in 0,1; j in 2,3.
        # Block A21: i in 2,3; j in 0,1.
        # Block A22: i in 2,3; j in 2,3.
        
        # For each of the 7 top-level products, we have a linear combination of blocks.
        # We need to generate U, V, W vectors of length 16.
        
        U_2, V_2, W_2 = strassen_2x2()
        
        # Top-level Strassen coefficients for A blocks
        # M1: A11 + A22
        # M2: A21 + A22
        # M3: A11
        # M4: A22
        # M5: A11 + A12
        # M6: A21 - A11
        # M7: A12 - A22
        
        # Coefficients for B blocks
        # M1: B11 + B22
        # M2: B11
        # M3: B12 - B22
        # M4: B21 - B11
        # M5: B22
        # M6: B11 + B12
        # M7: B21 + B22
        
        # Coefficients for C blocks
        # M1: C11 + C22
        # M2: C21 + C22
        # M3: C12 + C22
        # M4: C11 + C21
        # M5: C11 - C12
        # M6: C22
        # M7: C11
        
        # We will build U, V, W of shape (16, 49)
        U_total = [[0]*49 for _ in range(16)]
        V_total = [[0]*49 for _ in range(16)]
        W_total = [[0]*49 for _ in range(16)]
        
        # Block index mapping
        # Block 0: A11 (rows 0-1, cols 0-1)
        # Block 1: A12 (rows 0-1, cols 2-3)
        # Block 2: A21 (rows 2-3, cols 0-1)
        # Block 3: A22 (rows 2-3, cols 2-3)
        
        def get_block_indices(block_id):
            # Returns list of 4 global indices for a 2x2 block
            if block_id == 0: return [0, 1, 4, 5]
            if block_id == 1: return [2, 3, 6, 7]
            if block_id == 2: return [8, 9, 12, 13]
            if block_id == 3: return [10, 11, 14, 15]
            
        # Top-level coefficients
        # A_coeffs[prod_id][block_id]
        A_coeffs = [
            [1, 0, 0, 1], # M1
            [0, 0, 1, 1], # M2
            [1, 0, 0, 0], # M3
            [0, 0, 0, 1], # M4
            [1, 1, 0, 0], # M5
            [-1, 0, 1, 0], # M6
            [0, 1, 0, -1]  # M7
        ]
        
        B_coeffs = [
            [1, 0, 0, 1], # M1
            [1, 0, 0, 0], # M2
            [0, 1, 0, -1], # M3
            [0, 0, 1, -1], # M4
            [0, 0, 0, 1], # M5
            [1, 1, 0, 0], # M6
            [0, 0, 1, 1]  # M7
        ]
        
        C_coeffs = [
            [1, 0, 0, 1], # M1
            [0, 0, 1, 1], # M2
            [0, 1, 0, 1], # M3
            [1, 0, 1, 0], # M4
            [1, -1, 0, 0], # M5
            [0, 0, 0, 1], # M6
            [1, 0, 0, 0]  # M7
        ]
        
        # For each top-level product (0..6)
        for p_id in range(7):
            # Get local U, V, W for 2x2 multiplication (7 terms)
            # These are 4x7 matrices
            
            # For each local term (0..6)
            for l_id in range(7):
                col_idx = p_id * 7 + l_id
                
                # Construct U vector for this term
                # U is linear combo of A blocks
                # U_global[i] = sum_{b} A_coeffs[p_id][b] * U_2_local[i_local]
                # where i_local is index in 2x2 block b
                
                # We need to map local 2x2 indices (0,1,2,3) to global indices
                # Local index 0 -> block row 0, col 0
                # Local index 1 -> block row 0, col 1
                # Local index 2 -> block row 1, col 0
                # Local index 3 -> block row 1, col 1
                
                for b in range(4):
                    coeff = A_coeffs[p_id][b]
                    if coeff == 0: continue
                    
                    idxs = get_block_indices(b)
                    # idxs[0] is local 0, idxs[1] is local 1, etc.
                    
                    # U_2 is 4x7. U_2[local_idx][l_id]
                    for l_idx in range(4):
                        global_idx = idxs[l_idx]
                        val = coeff * U_2[l_idx][l_id]
                        U_total[global_idx][col_idx] = val
                        
                # Construct V vector
                for b in range(4):
                    coeff = B_coeffs[p_id][b]
                    if coeff == 0: continue
                    
                    idxs = get_block_indices(b)
                    for l_idx in range(4):
                        global_idx = idxs[l_idx]
                        val = coeff * V_2[l_idx][l_id]
                        V_total[global_idx][col_idx] = val
                        
                # Construct W vector
                for b in range(4):
                    coeff = C_coeffs[p_id][b]
                    if coeff == 0: continue
                    
                    idxs = get_block_indices(b)
                    for l_idx in range(4):
                        global_idx = idxs[l_idx]
                        val = coeff * W_2[l_idx][l_id]
                        W_total[global_idx][col_idx] = val
                        
        return U_total, V_total, W_total

    # Generate sub-decompositions
    U_444, V_444, W_444 = strassen_4x4()
    
    # 4x4x1 Schoolbook
    U_441, V_441, W_441 = schoolbook(4, 4, 1)
    
    # 4x1x4 Schoolbook
    U_414, V_414, W_414 = schoolbook(4, 1, 4)
    
    # 4x1x1 Schoolbook
    U_411, V_411, W_411 = schoolbook(4, 1, 1)
    
    # Total Rank
    rank_444 = 49
    rank_441 = 16
    rank_414 = 16
    rank_411 = 4
    total_rank = rank_444 + rank_441 + rank_414 + rank_411
    
    # Initialize Global Matrices
    U = [[0] * total_rank for _ in range(20)]
    V = [[0] * total_rank for _ in range(25)]
    W = [[0] * total_rank for _ in range(20)]
    
    # Fill 4x4x4 part
    # A indices: i in 0..3, j in 0..3 -> i*5 + j
    # B indices: j in 0..3, k in 0..3 -> j*5 + k
    # C indices: k in 0..3, i in 0..3 -> k*4 + i
    
    for i in range(4):
        for j in range(4):
            for c in range(rank_444):
                U[i*5 + j][c] = U_444[i*4 + j][c]
                
    for j in range(4):
        for k in range(4):
            for c in range(rank_444):
                V[j*5 + k][c] = V_444[j*4 + k][c]
                
    for k in range(4):
        for i in range(4):
            for c in range(rank_444):
                W[k*4 + i][c] = W_444[k*4 + i][c]
                
    # Fill 4x4x1 part
    offset = rank_444
    # A: i in 0..3, j in 0..3
    # B: j in 0..3, k=4
    # C: k=4, i in 0..3
    
    for i in range(4):
        for j in range(4):
            for c in range(rank_441):
                U[i*5 + j][offset + c] = U_441[i*4 + j][c]
                
    for j in range(4):
        for c in range(rank_441):
            V[j*5 + 4][offset + c] = V_441[j*4][c] # V is 4x4 in local, m=4, p=1
            
    for i in range(4):
        for c in range(rank_441):
            W[4*4 + i][offset + c] = W_441[i][c] # W is 1x4 in local, p=1, n=4. Index k*n+i = 0*4+i = i. Wait.
            # Local W for 4x4x1: p=1, n=4. Size 1x4.
            # Global W index: k=4, i in 0..3. Index 4*4 + i.
            # Local W index: k=0, i in 0..3. Index 0*4 + i = i.
            # So W_441[i][c] maps to W[4*4+i][offset+c].
            
    # Fill 4x1x4 part
    offset += rank_441
    # A: i in 0..3, j=4
    # B: j in 0..3, k in 0..3
    # C: k in 0..3, i in 0..3
    
    for i in range(4):
        for c in range(rank_414):
            U[i*5 + 4][offset + c] = U_414[i][c] # Local A is 4x1. Index i.
            
    for j in range(4):
        for k in range(4):
            for c in range(rank_414):
                V[j*5 + k][offset + c] = V_414[j*4 + k][c] # Local B is 4x4.
                
    for k in range(4):
        for i in range(4):
            for c in range(rank_414):
                W[k*4 + i][offset + c] = W_414[k*4 + i][c] # Local C is 4x4.
                
    # Fill 4x1x1 part
    offset += rank_414
    # A: i in 0..3, j=4
    # B: j=4, k=4
    # C: k=4, i in 0..3
    
    for i in range(4):
        for c in range(rank_411):
            U[i*5 + 4][offset + c] = U_411[i][c]
            
    for c in range(rank_411):
        V[4*5 + 4][offset + c] = V_411[0][c] # Local B is 1x1.
        
    for i in range(4):
        for c in range(rank_411):
            W[4*4 + i][offset + c] = W_411[i][c] # Local C is 1x4? No, 4x1x1. C is 1x4?
            # p=1, n=4. C size 1x4.
            # Global C index: k=4, i.
            # Local C index: k=0, i.
            # W_411 is 4x4? No, schoolbook(4,1,1) returns W of size (1*4, 4) = (4,4).
            # W_411[i][c] is correct.

    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
