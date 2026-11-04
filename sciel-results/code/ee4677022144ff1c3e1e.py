"""Improved program for matrix multiplication 4x5x6 rank minimization."""
DIMENSIONS = [4, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = 4, 5, 6
    
    # Strassen decomposition for 2x2x2
    # M_{2,2,2}: rank 7
    # U: (4, 7), V: (4, 7), W: (4, 7)
    # Rows of U: (a11, a12, a21, a22) indexed as i*m+j = i*2+j
    # Rows of V: (b11, b12, b21, b22) indexed as j*p+k = j*2+k
    # Rows of W: (c11, c12, c21, c22) indexed as k*n+i = k*2+i
    
    # Strassen formula:
    # M1 = (a11+a22)(b11+b22)
    # M2 = (a21+a22)b11
    # M3 = a11(b12-b22)
    # M4 = a22(b21-b11)
    # M5 = (a11+a12)b22
    # M6 = (a21-a11)(b11+b12)
    # M7 = (a12-a22)(b21+b22)
    #
    # c11 = M1 + M4 - M5 + M7
    # c12 = M3 + M5
    # c21 = M2 + M4
    # c22 = M1 - M2 + M3 + M6
    
    U2 = [[1, 1, 0, 0, 1, -1, 0],
          [0, 0, 1, 0, 1, 0, 1],
          [0, 1, 0, 0, 0, 1, 0],
          [1, 1, 0, 1, 0, 0, -1]]
    V2 = [[1, 1, 0, -1, 0, 1, 0],
          [0, 0, 1, 0, 0, 1, 0],
          [0, 0, 0, 1, 0, 0, 1],
          [1, 0, -1, 0, 1, 0, 1]]
    W2 = [[1, 0, 0, 1, -1, 0, 1],
          [0, 0, 1, 0, 1, 0, 0],
          [0, 1, 0, 1, 0, 0, 0],
          [1, -1, 1, 0, 0, 1, 0]]
    
    # Verify: for each (i,j,k), sum_r U[i*2+j,r]*V[j*2+k,r]*W[k*2+i,r] should be 1
    # Let's check: U[row_i, r] * V[row_j, r] * W[row_k, r] where row_i = i*2+j, row_j = j*2+k, row_k = k*2+i
    
    # Now build 4x4x4 Strassen from 2x2x2 Strassen
    # 4x4 matrices are 2x2 blocks of 2x2 matrices
    # The 4x4x4 tensor is composed of 8 tensor products of 2x2x2 tensors
    # Actually, for block Strassen, we treat 4x4 as 2x2 blocks of 2x2
    # The multiplication of two 4x4 matrices C = AB where A,B are 4x4
    # Partition A into blocks A11, A12, A21, A22 (each 2x2)
    # Partition B into blocks B11, B12, B21, B22 (each 2x2)
    # Then C11 = A11*B11 + A12*B21, C12 = A11*B12 + A12*B22
    # C21 = A21*B11 + A22*B21, C22 = A21*B12 + A22*B22
    # Using Strassen on the block level:
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
    #
    # Each Mi is a 2x2 matrix product, which can be done with rank 7
    # So total rank = 7 * 7 = 49
    
    # For 4x4x4, the indices are:
    # A is 4x4, indexed by (i,j) with i in [0,4), j in [0,4)
    # B is 4x4, indexed by (j,k) with j in [0,4), k in [0,4)
    # C is 4x4, indexed by (k,i) with k in [0,4), i in [0,4)
    # 
    # Block structure:
    # A11 = A[0:2, 0:2], A12 = A[0:2, 2:4], A21 = A[2:4, 0:2], A22 = A[2:4, 2:4]
    # B11 = B[0:2, 0:2], B12 = B[0:2, 2:4], B21 = B[2:4, 0:2], B22 = B[2:4, 2:4]
    #
    # In terms of individual elements:
    # A[i,j] where i = block_i*2 + local_i, j = block_j*2 + local_j
    # B[j,k] where j = block_j*2 + local_j, k = block_k*2 + local_k
    # C[k,i] where k = block_k*2 + local_k, i = block_i*2 + local_i
    #
    # For each Strassen term Mi (block level, i=1..7), we need to compute a 2x2 matrix product
    # Mi = Ui * Vi where Ui is a 2x2 matrix formed from A blocks, Vi from B blocks
    # Then C gets contribution from Mi * Wi where Wi is a 2x2 matrix of coefficients
    #
    # Let's build the 4x4x4 decomposition:
    # For each block-level Strassen term r in range(7):
    #   For each local 2x2x2 Strassen term s in range(7):
    #     This gives one scalar multiplication
    #     The U factor: product of U4_block[r] and U2[s]
    #     The V factor: product of V4_block[r] and V2[s]
    #     The W factor: product of W4_block[r] and W2[s]
    #
    # U4_block[r]: maps A elements to the r-th Strassen block term
    # For block Strassen:
    # M1: (A11+A22)(B11+B22)
    #   A part: A11 + A22, B part: B11 + B22
    # M2: (A21+A22)B11
    #   A part: A21+A22, B part: B11
    # M3: A11(B12-B22)
    #   A part: A11, B part: B12-B22
    # M4: A22(B21-B11)
    #   A part: A22, B part: B21-B11
    # M5: (A11+A12)B22
    #   A part: A11+A12, B part: B22
    # M6: (A21-A11)(B11+B12)
    #   A part: A21-A11, B part: B11+B12
    # M7: (A12-A22)(B21+B22)
    #   A part: A12-A22, B part: B21+B22
    
    # U4_block[r] is a 4x4 matrix (for A's 4x4), but we only need the relevant parts
    # Actually, U4_block[r] has nonzero entries only in certain 2x2 blocks of the 4x4
    # Let's define U4_block as 4x4 (rows are A's row*4+col index)
    # A[i,j] where i in [0,4), j in [0,4)
    # Block (bi, bj) contains A[bi*2+li, bj*2+lj]
    
    U4_block = []
    V4_block = []
    # Coefficients for W4_block (how each Mi contributes to C's blocks)
    W4_block = []  # Each entry is a 4x4 matrix of coefficients
    
    # M1: (A11+A22)(B11+B22)
    U4_block.append([
        [1, 1, 0, 0],  # A11
        [1, 1, 0, 0],  # A11
        [0, 0, 1, 1],  # A22
        [0, 0, 1, 1]   # A22
    ])
    V4_block.append([
        [1, 1, 0, 0],  # B11
        [1, 1, 0, 0],  # B11
        [0, 0, 1, 1],  # B22
        [0, 0, 1, 1]   # B22
    ])
    # C11 = M1 + M4 - M5 + M7, C12 = M3 + M5, C21 = M2 + M4, C22 = M1 - M2 + M3 + M6
    # M1 contributes to C11 and C22
    W4_block.append([
        [1, 0, 0, 0],  # C11 (block (0,0))
        [0, 0, 0, 0],  # C12 (block (0,1))
        [0, 0, 0, 0],  # C21 (block (1,0))
        [0, 1, 0, 0],  # C22 (block (1,1))
    ])
    # Wait, W4_block should be structured as C[k,i] where k is row, i is col
    # C is 4x4, blocks are C[bi*k_block, bi*i_block]
    # Let me re-index: C[k,i], block (bk, bi) where bk in {0,1}, bi in {0,1}
    # C11 = C[0:2, 0:2] = block (0,0)
    # C12 = C[0:2, 2:4] = block (0,1)
    # C21 = C[2:4, 0:2] = block (1,0)
    # C22 = C[2:4, 2:4] = block (1,1)
    
    W4_block = []
    # M1: contributes to C11 (+1) and C22 (+1)
    W4_block.append([
        [1, 0, 0, 0],  # C11
        [0, 0, 0, 0],  # C12
        [0, 0, 0, 0],  # C21
        [0, 0, 1, 0],  # C22 - wait, this is wrong
    ])
    
    # Let me be more careful. W4_block[r] should be a 4x4 matrix where
    # W4_block[r][k*4+i] is the coefficient for C[k,i] from the r-th block Strassen term
    # But actually, each Mi is a 2x2 matrix product, so it produces a 2x2 result
    # which gets placed in specific blocks of C
    
    # Let me redo this. For block Strassen term r, the result is a 2x2 matrix
    # that contributes to C blocks according to the W4_block pattern.
    # Actually, the W part for 4x4x4 needs to map from (local_k, local_i) within a block
    # to the global C[k,i].
    
    # I think the cleanest way is to just build the full 49-term decomposition directly.
    # For each block Strassen term r (0..6) and local Strassen term s (0..6):
    #   U factor: U4_block[r] (4x4) tensor U2[s] (2x2) -> 4x4 matrix
    #   V factor: V4_block[r] (4x4) tensor V2[s] (2x2) -> 4x4 matrix  
    #   W factor: W4_block[r] (4x4) tensor W2[s] (2x2) -> 4x4 matrix
    #
    # But W4_block[r] is not just a 4x4 matrix; it's a pattern of which blocks of C
    # the 2x2 result from term r goes into.
    
    # Let me define it differently. For block term r:
    # The 2x2 result M_r is added to specific 2x2 blocks of C with certain coefficients.
    # Let's say M_r contributes coeff[r][block] * (2x2 result) to C[block]
    # where block is one of (0,0), (0,1), (1,0), (1,1)
    
    # From Strassen:
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    # So for block term r:
    # r=0 (M1): C11 += 1*M1, C22 += 1*M1
    # r=1 (M2): C21 += 1*M2, C22 += -1*M2
    # r=2 (M3): C12 += 1*M3, C22 += 1*M3
    # r=3 (M4): C11 += 1*M4, C21 += 1*M4
    # r=4 (M5): C11 += -1*M5, C12 += 1*M5
    # r=5 (M6): C22 += 1*M6
    # r=6 (M7): C11 += 1*M7
    
    block_to_c = {
        0: [(0, 1), (3, 1)],   # M1: C11(+1), C22(+1)  -- block indices: C11=block(0,0), C22=block(1,1)
        1: [(2, 1), (3, -1)],  # M2: C21(+1), C22(-1)  -- C21=block(1,0), C22=block(1,1)
        2: [(1, 1), (3, 1)],   # M3: C12(+1), C22(+1)  -- C12=block(0,1), C22=block(1,1)
        3: [(0, 1), (2, 1)],   # M4: C11(+1), C21(+1)
        4: [(0, -1), (1, 1)],  # M5: C11(-1), C12(+1)
        5: [(3, 1)],           # M6: C22(+1)
        6: [(0, 1)],           # M7: C11(+1)
    }
    # Block index: (block_row, block_col) where block_row in {0,1}, block_col in {0,1}
    # block(0,0) = C11, block(0,1) = C12, block(1,0) = C21, block(1,1) = C22
    # Let me use a flat index: block_idx = block_row*2 + block_col
    # block(0,0)=0, block(0,1)=1, block(1,0)=2, block(1,1)=3
    
    block_contributions = {
        0: [(0, 1), (3, 1)],
        1: [(2, 1), (3, -1)],
        2: [(1, 1), (3, 1)],
        3: [(0, 1), (2, 1)],
        4: [(0, -1), (1, 1)],
        5: [(3, 1)],
        6: [(0, 1)],
    }
    
    # Now build 4x4x4 with rank 49
    rank_444 = 49
    U444 = [[0] * rank_444 for _ in range(16)]  # 4*4 = 16 rows
    V444 = [[0] * rank_444 for _ in range(16)]  # 4*4 = 16 rows
    W444 = [[0] * rank_444 for _ in range(16)]  # 4*4 = 16 rows
    
    col = 0
    for r in range(7):  # block Strassen term
        for s in range(7):  # local Strassen term
            # U factor for A
            # U4_block[r] is 4x4, U2[s] is 2x2
            # The combined U is 4x4 where U_combined[i,j] = sum over (bi,bj) of U4_block[r][i,j] * ...
            # Actually, U4_block[r] already specifies which elements of A are used
            # U4_block[r] is a 4x4 matrix, and U2[s] is 2x2
            # The tensor product is: for A[i,j], the coefficient is U4_block[r][i,j] * U2[local_i, local_j]
            # where local_i = i % 2, local_j = j % 2
            for i in range(4):
                for j in range(4):
                    local_i = i % 2
                    local_j = j % 2
                    coeff = U4_block[r][i][j] * U2[local_i][local_j]
                    if coeff != 0:
                        U444[i * 4 + j][col] = coeff
            
            # V factor for B
            for j in range(4):
                for k in range(4):
                    local_j = j % 2
                    local_k = k % 2
                    coeff = V4_block[r][j][k] * V2[local_j][local_k]
                    if coeff != 0:
                        V444[j * 4 + k][col] = coeff
            
            # W factor for C
            # For each block (bk, bi) that this term contributes to, with coefficient c:
            # C[k,i] where k = bk*2 + local_k, i = bi*2 + local_i
            # The coefficient is c * W2[local_k][local_i]
            for (block_idx, c) in block_contributions[r]:
                bk = block_idx // 2
                bi = block_idx % 2
                for local_k in range(2):
                    for local_i in range(2):
                        k = bk * 2 + local_k
                        i = bi * 2 + local_i
                        coeff = c * W2[local_k][local_i]
                        if coeff != 0:
                            W444[k * 4 + i][col] = coeff
            
            col += 1
    
    # Now build the full 4x5x6 decomposition
    # We partition:
    # A: 4x5, B: 5x6, C: 6x4 (output indexed as W[k*n+i] = W[k*4+i])
    # 
    # Blocks:
    # A[0:4, 0:4] * B[0:4, 0:6] -> C[0:4, 0:6] (but C is 6x4, so C[k,i] for k in [0,6), i in [0,4))
    # Actually let me re-read the problem.
    # "multiplying a 4 by 5 matrix by a 5 by 6 matrix"
    # A is 4x5, B is 5x6, result C is 4x6
    # But the output is W[k*n+i,r] where n=4, so W has shape (p*n, rank) = (6*4, rank) = (24, rank)
    # This suggests C is indexed as C[k, i] where k in [0,6), i in [0,4)
    # So C is 6x4, which means C = B * A^T or something?
    # 
    # Wait, let me re-read: "The output factor follows the official trace convention W[k*n+i,r]."
    # For matrix multiplication A (n x m) * B (m x p) = C (n x p)
    # But here n=4, m=5, p=6
    # C should be 4x6
    # But W[k*n+i] with n=4 means W index = k*4+i, so k in [0,6), i in [0,4)
    # This means C is indexed as C[k, i] where k is the "column" of B (p=6) and i is the "row" of A (n=4)
    # So C[k, i] = sum_j B[j, k] * A[i, j] = (B * A^T)[k, i]
    # Or equivalently, C = B @ A^T, which is 6x4
    # 
    # So the tensor is: C[k, i] = sum_{j=0}^{4} A[i, j] * B[j, k]
    # U: shape (n*m, rank) = (4*5, rank) = (20, rank), indexed by (i, j) -> i*5+j
    # V: shape (m*p, rank) = (5*6, rank) = (30, rank), indexed by (j, k) -> j*6+k
    # W: shape (p*n, rank) = (6*4, rank) = (24, rank), indexed by (k, i) -> k*4+i
    
    # Now I need to decompose this 4x5x6 tensor.
    # I can use the 4x4x4 Strassen decomposition for the top-left 4x4 block of A and B,
    # and handle the remaining parts with schoolbook.
    
    # A is 4x5, B is 5x6
    # Partition j in [0,5) into j in [0,4) and j in [4,5)
    # Partition k in [0,6) into k in [0,4) and k in [4,6)
    
    # Block 1: i in [0,4), j in [0,4), k in [0,4) -> use 4x4x4 Strassen (rank 49)
    # Block 2: i in [0,4), j in [0,4), k in [4,6) -> schoolbook 4x4x2 (rank 32)
    # Block 3: i in [0,4), j in [4,5), k in [0,4) -> schoolbook 4x1x4 (rank 16)
    # Block 4: i in [0,4), j in [4,5), k in [4,6) -> schoolbook 4x1x2 (rank 8)
    
    total_rank = 49 + 32 + 16 + 8  # = 105
    
    U = [[0] * total_rank for _ in range(20)]
    V = [[0] * total_rank for _ in range(30)]
    W = [[0] * total_rank for _ in range(24)]
    
    # Block 1: 4x4x4 Strassen
    col_offset = 0
    for i in range(4):
        for j in range(4):
            for r in range(49):
                U[i * 5 + j][col_offset + r] = U444[i * 4 + j][r]
    for j in range(4):
        for k in range(4):
            for r in range(49):
                V[j * 6 + k][col_offset + r] = V444[j * 4 + k][r]
    for k in range(4):
        for i in range(4):
            for r in range(49):
                W[k * 4 + i][col_offset + r] = W444[k * 4 + i][r]
    col_offset += 49
    
    # Block 2: schoolbook 4x4x2 (i in [0,4), j in [0,4), k in [4,6))
    col_offset += 32  # skip, we'll fill in
    
    # Block 3: schoolbook 4x1x4 (i in [0,4), j=4, k in [0,4))
    col_offset += 16
    
    # Block 4: schoolbook 4x1x2 (i in [0,4), j=4, k in [4,6))
    col_offset += 8
    
    # Now fill in blocks 2, 3, 4
    col = 49
    # Block 2: i in [0,4), j in [0,4), k in [4,6)
    for i in range(4):
        for j in range(4):
            for k in range(4, 6):
                U[i * 5 + j][col] = 1
                V[j * 6 + k][col] = 1
                W[k * 4 + i][col] = 1
                col += 1
    
    # Block 3: i in [0,4), j=4, k in [0,4)
    for i in range(4):
        for k in range(4):
            U[i * 5 + 4][col] = 1
            V[4 * 6 + k][col] = 1
            W[k * 4 + i][col] = 1
            col += 1
    
    # Block 4: i in [0,4), j=4, k in [4,6)
    for i in range(4):
        for k in range(4, 6):
            U[i * 5 + 4][col] = 1
            V[4 * 6 + k][col] = 1
            W[k * 4 + i][col] = 1
            col += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
