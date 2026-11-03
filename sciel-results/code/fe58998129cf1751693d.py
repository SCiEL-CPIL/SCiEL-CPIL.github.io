"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # n=4, m=4, p=8
    
    # We'll use Strassen's algorithm for 2x2x2 (rank 7) and compose it
    # For 4x4x4: treat as 2x2 blocks of 2x2, each 2x2x2 uses rank 7 -> total rank 7
    # For 4x4x8: two 4x4x4 multiplications, each rank 7 -> total rank 14
    
    # Build the 2x2x2 Strassen decomposition
    # Indices: A is 2x2, B is 2x2, C is 2x2
    # A indices: (i,j) with i,j in {0,1}
    # B indices: (j,k) with j,k in {0,1}
    # C indices: (i,k) with i,k in {0,1}
    
    # For 2x2x2 Strassen, rank 7:
    # U has 4 rows (A elements), V has 4 rows (B elements), W has 4 rows (C elements)
    # 7 columns (the 7 intermediate products)
    
    # Let's build it step by step
    # A11 = A[0,0], A12 = A[0,1], A21 = A[1,0], A22 = A[1,1]
    # B11 = B[0,0], B12 = B[0,1], B21 = B[1,0], B22 = B[1,1]
    # C11 = C[0,0], C12 = C[0,1], C21 = C[1,0], C22 = C[1,1]
    
    # Strassen:
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)(B11)
    # M3 = (A11)(B12-B22)
    # M4 = (A22)(B21-B11)
    # M5 = (A11+A12)(B22)
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    # U matrix: rows indexed by A element (i*2+j), columns by which M they contribute to
    # V matrix: rows indexed by B element (j*2+k), columns by which M they contribute to
    # W matrix: rows indexed by C element (i*2+k), columns by which M they contribute to
    
    # For each M, we need to specify which A and B elements go into it, and how the result contributes to C
    
    # Let's define the Strassen decomposition explicitly
    # Each M_t is a product of a linear combination of A elements and a linear combination of B elements
    
    # M1: A part = A11 + A22, B part = B11 + B22
    # M2: A part = A21 + A22, B part = B11
    # M3: A part = A11, B part = B12 - B22
    # M4: A part = A22, B part = B21 - B11
    # M5: A part = A11 + A12, B part = B22
    # M6: A part = A21 - A11, B part = B11 + B12
    # M7: A part = A12 - A22, B part = B21 + B22
    
    # U: A11=(0,0), A12=(0,1), A21=(1,0), A22=(1,1) -> indices 0,1,2,3
    # V: B11=(0,0), B12=(0,1), B21=(1,0), B22=(1,1) -> indices 0,1,2,3
    # W: C11=(0,0), C12=(0,1), C21=(1,0), C22=(1,1) -> indices 0,1,2,3
    
    # U[i,j] where i is A index, j is M index (0-6)
    U2 = [[0]*7 for _ in range(4)]
    U2[0][0] = 1  # A11 in M1
    U2[3][0] = 1  # A22 in M1
    U2[2][1] = 1  # A21 in M2
    U2[3][1] = 1  # A22 in M2
    U2[0][2] = 1  # A11 in M3
    U2[3][3] = 1  # A22 in M4
    U2[0][4] = 1  # A11 in M5
    U2[1][4] = 1  # A12 in M5
    U2[2][5] = 1  # A21 in M6
    U2[0][5] = -1 # A11 in M6
    U2[1][6] = 1  # A12 in M7
    U2[3][6] = -1 # A22 in M7
    
    V2 = [[0]*7 for _ in range(4)]
    V2[0][0] = 1  # B11 in M1
    V2[3][0] = 1  # B22 in M1
    V2[0][1] = 1  # B11 in M2
    V2[1][2] = 1  # B12 in M3
    V2[3][2] = -1 # B22 in M3
    V2[2][3] = 1  # B21 in M4
    V2[0][3] = -1 # B11 in M4
    V2[3][4] = 1  # B22 in M5
    V2[0][5] = 1  # B11 in M6
    V2[1][5] = 1  # B12 in M6
    V2[2][6] = 1  # B21 in M7
    V2[3][6] = 1  # B22 in M7
    
    W2 = [[0]*7 for _ in range(4)]
    # C11 = M1 + M4 - M5 + M7
    W2[0][0] = 1  # C11 from M1
    W2[0][3] = 1  # C11 from M4
    W2[0][4] = -1 # C11 from M5
    W2[0][6] = 1  # C11 from M7
    # C12 = M3 + M5
    W2[1][2] = 1  # C12 from M3
    W2[1][4] = 1  # C12 from M5
    # C21 = M2 + M4
    W2[2][1] = 1  # C21 from M2
    W2[2][3] = 1  # C21 from M4
    # C22 = M1 - M2 + M3 + M6
    W2[3][0] = 1  # C22 from M1
    W2[3][1] = -1 # C22 from M2
    W2[3][2] = 1  # C22 from M3
    W2[3][5] = 1  # C22 from M6
    
    # Now compose for 4x4x4 using 2x2 blocks of 2x2
    # A is 4x4, viewed as 2x2 blocks of 2x2
    # Block (bi,bj) of A has elements A[2*bi+di, 2*bj+dj]
    # Similarly for B and C
    
    # For 4x4x4 with 2x2 blocks:
    # A block (bi,bj): rows 2*bi to 2*bi+1, cols 2*bj to 2*bj+1
    # Within each block, local indices di,dj in {0,1}
    
    # The 4x4x4 tensor can be decomposed using the 2x2x2 Strassen at the block level
    # But we also need to handle the within-block structure
    
    # Actually, for 4x4x4, we can use a single level of Strassen where we treat
    # the 4x4 matrices as 2x2 blocks of 2x2 matrices. This gives rank 7.
    
    # U for 4x4x4: 16 rows (4x4 A elements), 7 columns
    # Row index: (bi, di, bj, dj) -> let's use (bi*2+di)*4 + (bj*2+dj) = bi*8 + di*4 + bj*2 + dj
    # Actually, let's use row = (i, j) where i,j in 0..3, index = i*4+j
    
    # For block (bi,bj) of A, local element (di,dj) is at global position (2*bi+di, 2*bj+dj)
    # Global index = (2*bi+di)*4 + (2*bj+dj)
    
    # Similarly for B: block (bj,bk), local (dj,dk) at global (2*bj+dj, 2*bk+dk)
    # Global index = (2*bj+dj)*4 + (2*bk+dk)
    
    # For C: block (bi,bk), local (di,dk) at global (2*bi+di, 2*bk+dk)
    # Global index = (2*bi+di)*4 + (2*bk+dk)
    
    # But wait - the output convention is W[k*n+i, r] where n=4
    # So C element (i,k) is at row k*4+i in W
    
    # Let me rebuild with proper indexing
    
    # For 4x4x4:
    # U: 16 rows (i*4+j for A[i,j]), 7 cols
    # V: 16 rows (j*4+k for B[j,k]), 7 cols  
    # W: 16 rows (k*4+i for C[i,k]), 7 cols
    
    U4 = [[0]*7 for _ in range(16)]
    V4 = [[0]*7 for _ in range(16)]
    W4 = [[0]*7 for _ in range(16)]
    
    # For each block-level Strassen term t (0-6):
    # A part: sum over bi,dj of U2[bi*2+di][t] * A[2*bi+di, 2*bj+dj]... 
    # Wait, I need to be more careful.
    
    # The block-level Strassen operates on 2x2 blocks. Each block is a 2x2 matrix.
    # The product of two 2x2 blocks is computed using the 7-term Strassen.
    # But actually, for the full 4x4x4 tensor, we can't just do block-level Strassen
    # because the within-block structure matters.
    
    # Let me think again. The 4x4x4 tensor T[i,j,k] = delta(i,j,k) where the tensor
    # is defined as T[i,j,k] = 1 if we're computing C[i,k] = sum_j A[i,j]*B[j,k]
    # So T[i,j,k] = 1 for all i,j,k (it's the identity tensor for matmul)
    
    # Actually no. The matrix multiplication tensor has T[i,j,k] = 1 for all i,j,k.
    # The rank is the minimum number of rank-1 tensors that sum to T.
    
    # For 2x2x2, the tensor is all ones (8 elements), and Strassen shows rank 7.
    # For 4x4x4, the tensor is all ones (64 elements).
    
    # A simple construction: use the 2x2x2 Strassen decomposition at each of the
    # 8 "sub-tensors" formed by fixing the high-order bits. But that gives rank 8*7=56.
    
    # Better: use Kronecker product structure. If we have a rank-r decomposition of
    # 2x2x2, we can get a rank-r decomposition of 4x4x4 by... actually no, that's not
    # straightforward.
    
    # Let me just use the straightforward approach: for 4x4x4, use the fact that
    # we can decompose it using 7 terms at the block level, but each term involves
    # products of 2x2 blocks, which themselves need decomposition.
    
    # Actually, the cleanest approach: 4x4x4 can be seen as a 2x2 block matrix where
    # each block is 2x2. The multiplication of two 4x4 matrices (as 2x2 block matrices)
    # involves 8 block multiplications (each 2x2x2) in the schoolbook way, or 7 using
    # Strassen at the block level. But each block multiplication is a 2x2x2 tensor.
    
    # So if I use Strassen at the block level (7 terms), and for each term I need to
    # multiply two 2x2 blocks, I can use the 2x2x2 Strassen decomposition for each.
    # But that gives 7*7 = 49 terms total.
    
    # That matches the "known minimal rank of 49" from the knowledge base!
    
    # So: 4x4x4 has rank at most 49, achieved by 2-level Strassen.
    
    # Let me construct this:
    # Level 1: Block-level Strassen (7 terms)
    # Level 2: For each block-level term, do a 2x2x2 Strassen (7 terms each)
    # Total: 49 terms
    
    # Block-level Strassen for 4x4x4:
    # A is 4x4, viewed as 2x2 blocks of 2x2:
    # A = [[A00, A01], [A10, A11]] where Aij is 2x2
    # B = [[B00, B01], [B10, B11]]
    # C = [[C00, C01], [C10, C11]]
    
    # Strassen:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B01-B11)
    # M4 = A11(B10-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # Now, each M_t is a product of a 2x2 matrix (linear combo of A blocks) and a 2x2 matrix
    # (linear combo of B blocks). Each such product is a 2x2x2 tensor.
    
    # For each t, I need to decompose the 2x2x2 product using Strassen (rank 7).
    
    # Let me index the 49 terms as (t, s) where t is the block-level term (0-6) and
    # s is the within-block Strassen term (0-6).
    
    # For term (t,s):
    # U part: from A, the coefficient for A[i,j] 
    # V part: from B, the coefficient for B[j,k]
    # W part: from C, the coefficient for C[i,k]
    
    # Let me build this carefully.
    
    # A block (bi,bj) has elements at positions (2*bi+di, 2*bj+dj) for di,dj in {0,1}
    # Global A index: (2*bi+di)*4 + (2*bj+dj)
    
    # B block (bj,bk) has elements at positions (2*bj+dj, 2*bk+dk) for dj,dk in {0,1}
    # Global B index: (2*bj+dj)*4 + (2*bk+dk)
    
    # C block (bi,bk) has elements at positions (2*bi+di, 2*bk+dk) for di,dk in {0,1}
    # Global C index: (2*bi+di)*4 + (2*bk+dk)
    # But output convention: W row = k*4+i, so C[i,k] is at row k*4+i
    # C element at (2*bi+di, 2*bk+dk) has i=2*bi+di, k=2*bk+dk
    # W row = (2*bk+dk)*4 + (2*bi+di)
    
    # Block-level Strassen coefficients:
    # For A blocks:
    # M1: A00 + A11 -> blocks (0,0) and (1,1) with coeff 1
    # M2: A10 + A11 -> blocks (1,0) and (1,1) with coeff 1
    # M3: A00 -> block (0,0) with coeff 1
    # M4: A11 -> block (1,1) with coeff 1
    # M5: A00 + A01 -> blocks (0,0) and (0,1) with coeff 1
    # M6: A10 - A00 -> blocks (1,0) with coeff 1, (0,0) with coeff -1
    # M7: A01 - A11 -> blocks (0,1) with coeff 1, (1,1) with coeff -1
    
    # For B blocks:
    # M1: B00 + B11 -> blocks (0,0) and (1,1) with coeff 1
    # M2: B00 -> block (0,0) with coeff 1
    # M3: B01 - B11 -> blocks (0,1) with coeff 1, (1,1) with coeff -1
    # M4: B10 - B00 -> blocks (1,0) with coeff 1, (0,0) with coeff -1
    # M5: B11 -> block (1,1) with coeff 1
    # M6: B00 + B01 -> blocks (0,0) and (0,1) with coeff 1
    # M7: B10 + B11 -> blocks (1,0) and (1,1) with coeff 1
    
    # For C blocks:
    # C00: M1 + M4 - M5 + M7
    # C01: M3 + M5
    # C10: M2 + M4
    # C11: M1 - M2 + M3 + M6
    
    # Within-block Strassen (2x2x2) for each block product:
    # The U2, V2, W2 matrices defined earlier
    
    # Now let me build the full 49-term decomposition for 4x4x4
    
    # A_block_coeffs[t][bi][bj] = coefficient of A block (bi,bj) in term t
    A_bc = [
        {(0,0):1, (1,1):1},  # M1
        {(1,0):1, (1,1):1},  # M2
        {(0,0):1},            # M3
        {(1,1):1},            # M4
        {(0,0):1, (0,1):1},  # M5
        {(1,0):1, (0,0):-1}, # M6
        {(0,1):1, (1,1):-1}, # M7
    ]
    
    B_bc = [
        {(0,0):1, (1,1):1},  # M1
        {(0,0):1},            # M2
        {(0,1):1, (1,1):-1}, # M3
        {(1,0):1, (0,0):-1}, # M4
        {(1,1):1},            # M5
        {(0,0):1, (0,1):1},  # M6
        {(1,0):1, (1,1):1},  # M7
    ]
    
    C_bc = [
        {(0,0):1, (1,0):1, (0,1):-1, (1,1):1},  # C00: M1+M4-M5+M7
        {(0,1):1, (0,0):1},                      # C01: M3+M5
        {(1,0):1, (1,1):1},                      # C10: M2+M4
        {(1,1):1, (1,0):-1, (0,1):1, (0,0):1},  # C11: M1-M2+M3+M6
    ]
    # Wait, C_bc should be indexed by C block, and for each C block, which M terms contribute
    # Let me redo: C_bc[ci][t] = coefficient
    C_bc = [
        {0:1, 3:1, 4:-1, 6:1},   # C00
        {2:1, 4:1},               # C01
        {1:1, 3:1},               # C10
        {0:1, 1:-1, 2:1, 5:1},   # C11
    ]
    
    # Now build U4, V4, W4 with 49 columns
    rank4 = 49
    U4 = [[0]*rank4 for _ in range(16)]
    V4 = [[0]*rank4 for _ in range(16)]
    W4 = [[0]*rank4 for _ in range(16)]
    
    for t in range(7):
        for s in range(7):
            col = t*7 + s
            
            # U part: for each A element (i,j), determine coefficient
            # A[i,j] is in block (bi,bj) where bi=i//2, bj=j//2, with local (di,dj)=(i%2,j%2)
            # For term t, block (bi,bj) has coefficient A_bc[t].get((bi,bj), 0)
            # Within the block, the local element (di,dj) has coefficient U2[di*2+dj][s]
            for i in range(4):
                for j in range(4):
                    bi, bj = i//2, j//2
                    di, dj = i%2, j%2
                    block_coeff = A_bc[t].get((bi,bj), 0)
                    local_coeff = U2[di*2+dj][s]
                    U4[i*4+j][col] += block_coeff * local_coeff
            
            # V part: for each B element (j,k)
            for j in range(4):
                for k in range(4):
                    bj, bk = j//2, k//2
                    dj, dk = j%2, k%2
                    block_coeff = B_bc[t].get((bj,bk), 0)
                    local_coeff = V2[dj*2+dk][s]
                    V4[j*4+k][col] += block_coeff * local_coeff
            
            # W part: for each C element (i,k), output row = k*4+i
            for i in range(4):
                for k in range(4):
                    bi, bk = i//2, k//2
                    di, dk = i%2, k%2
                    # C block (bi,bk) gets contribution from term t with coefficient C_bc[bi*2+bk][t]
                    c_block_idx = bi*2+bk
                    block_coeff = C_bc[c_block_idx].get(t, 0)
                    local_coeff = W2[di*2+dk][s]
                    W4[k*4+i][col] += block_coeff * local_coeff
    
    # Now for 4x4x8: split B into two 4x4 blocks (columns 0-3 and 4-7)
    # B is 4x8, so B[j,k] for j in 0..3, k in 0..7
    # Split into B_left = B[:,0:4] and B_right = B[:,4:8]
    # C is 4x8, C[i,k] for i in 0..3, k in 0..7
    # C_left = C[:,0:4] and C_right = C[:,4:8]
    
    # U is the same for both (4x4 A matrix)
    # V is 4x8, split into two 4x4 blocks
    # W is 8x4 (output convention: W[k*4+i]), split into two 4x4 blocks
    
    # Total rank: 49 + 49 = 98
    
    rank_total = 98
    U = [[0]*rank_total for _ in range(16)]  # 4*4 = 16 rows
    V = [[0]*rank_total for _ in range(32)]  # 4*8 = 32 rows
    W = [[0]*rank_total for _ in range(32)]  # 8*4 = 32 rows
    
    # First 49 columns: for B_left (k=0..3)
    for col in range(49):
        for row in range(16):
            U[row][col] = U4[row][col]
        for row in range(16):
            V[row][col] = V4[row][col]  # B_left[j,k] at row j*4+k
        for row in range(16):
            W[row][col] = W4[row][col]  # C_left[i,k] at row k*4+i
    
    # Next 49 columns: for B_right (k=4..7)
    for col in range(49):
        col2 = 49 + col
        for row in range(16):
            U[row][col2] = U4[row][col]
        for row in range(16):
            # B_right[j,k] where k=4..7, so B[j,4+k'] for k'=0..3
            # B row index: j*8 + (4+k')
            j, k_local = row // 4, row % 4
            V[j*8 + 4 + k_local][col2] = V4[row][col]
        for row in range(16):
            # C_right[i,k] where k=4..7, so C[i,4+k'] for k'=0..3
            # W row index: (4+k')*4 + i
            i, k_local = row % 4, row // 4  # Wait, W4 row is k*4+i
            # W4 row r corresponds to k=r//4, i=r%4
            k_local = row // 4
            i = row % 4
            W[(4+k_local)*4 + i][col2] = W4[row][col]
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
