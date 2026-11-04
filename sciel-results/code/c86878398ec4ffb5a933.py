"""Tensor decomposition for 4x4 matrix multiplication using Strassen's algorithm on 2x2 blocks."""
DIMENSIONS = [4, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # We'll use a block decomposition: partition 4x4 into 2x2 blocks.
    # There are 2x2 = 4 blocks of A, 4 blocks of B, and 4 blocks of C.
    # Each block multiplication is a 2x2x2 tensor.
    # Strassen's algorithm gives rank 7 for 2x2x2.
    # Total rank = 4 (block positions) * 7 (Strassen rank) = 28
    
    rank = 28
    U = [[0] * rank for _ in range(n * m)]  # 16 x 28
    V = [[0] * rank for _ in range(m * p)]  # 16 x 28
    W = [[0] * rank for _ in range(p * n)]  # 16 x 28
    
    # Strassen's 7 products for 2x2 matrices:
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
    
    # For each block position (bi, bj) of the 2x2 block grid:
    # A block at (bi, bj) corresponds to rows 2*bi to 2*bi+1, cols 2*bj to 2*bj+1 of A
    # B block at (bk, bl) corresponds to rows 2*bk to 2*bk+1, cols 2*bl to 2*bl+1 of B
    # C block at (ci, cj) corresponds to rows 2*ci to 2*ci+1, cols 2*cj to 2*cj+1 of C
    
    # For 4x4 matrices, the block index for element (i,j) is (i//2, j//2)
    # Local index within block: (i%2, j%2)
    
    # Let's define the Strassen matrices for 2x2:
    # U2[local_a_idx, s], V2[local_b_idx, s], W2[local_c_idx, s]
    # local_a_idx = 2*ai + aj where ai, aj in {0,1}
    # local_b_idx = 2*bi + bj where bi, bj in {0,1}
    # local_c_idx = 2*ci + cj where ci, cj in {0,1}
    
    # Strassen products (s=0..6):
    # Each product involves linear combinations of A entries, B entries, and C entries.
    
    # Let's build U2, V2, W2 for a single 2x2 block multiplication:
    # U2 has shape (4, 7), V2 has shape (4, 7), W2 has shape (4, 7)
    
    U2 = [[0]*7 for _ in range(4)]
    V2 = [[0]*7 for _ in range(4)]
    W2 = [[0]*7 for _ in range(4)]
    
    # M1 = (A00 + A11) * (B00 + B11)
    # A entries: A00=idx0, A01=idx1, A10=idx2, A11=idx3
    # B entries: B00=idx0, B01=idx1, B10=idx2, B11=idx3
    
    # s=0: M1
    U2[0][0] = 1  # A00
    U2[3][0] = 1  # A11
    V2[0][0] = 1  # B00
    V2[3][0] = 1  # B11
    # C00 += M1, C11 += M1
    W2[0][0] = 1  # C00
    W2[3][0] = 1  # C11
    
    # s=1: M2 = (A10 + A11) * B00
    U2[2][1] = 1  # A10
    U2[3][1] = 1  # A11
    V2[0][1] = 1  # B00
    # C10 += M2, C11 -= M2
    W2[2][1] = 1  # C10
    W2[3][1] = -1 # C11
    
    # s=2: M3 = A00 * (B01 - B11)
    U2[0][2] = 1  # A00
    V2[1][2] = 1  # B01
    V2[3][2] = -1 # B11
    # C01 += M3, C11 += M3
    W2[1][2] = 1  # C01
    W2[3][2] = 1  # C11
    
    # s=3: M4 = A11 * (B10 - B00)
    U2[3][3] = 1  # A11
    V2[2][3] = 1  # B10
    V2[0][3] = -1 # B00
    # C00 += M4, C10 += M4
    W2[0][3] = 1  # C00
    W2[2][3] = 1  # C10
    
    # s=4: M5 = (A00 + A01) * B11
    U2[0][4] = 1  # A00
    U2[1][4] = 1  # A01
    V2[3][4] = 1  # B11
    # C00 -= M5, C01 += M5
    W2[0][4] = -1 # C00
    W2[1][4] = 1  # C01
    
    # s=5: M6 = (A10 - A00) * (B00 + B01)
    U2[2][5] = 1  # A10
    U2[0][5] = -1 # A00
    V2[0][5] = 1  # B00
    V2[1][5] = 1  # B01
    # C11 += M6
    W2[3][5] = 1  # C11
    
    # s=6: M7 = (A01 - A11) * (B10 + B11)
    U2[1][6] = 1  # A01
    U2[3][6] = -1 # A11
    V2[2][6] = 1  # B10
    V2[3][6] = 1  # B11
    # C00 += M7, C11 += M7
    W2[0][6] = 1  # C00
    W2[3][6] = 0  # C11
    
    # Now we need to place these into the global U, V, W matrices.
    # The 4x4 matrix is partitioned into 2x2 blocks.
    # Block (bi, bj) of A corresponds to global rows 2*bi to 2*bi+1, cols 2*bj to 2*bj+1.
    # For the tensor decomposition, we treat each block multiplication independently.
    # But actually, the blocks interact: C block (ci, cj) = sum over k of A block (ci, k) * B block (k, cj)
    
    # Wait, I need to be more careful. Let me reconsider.
    
    # Actually, for 4x4 matrix multiplication, if we partition into 2x2 blocks:
    # A has blocks A00, A01, A10, A11 (each 2x2)
    # B has blocks B00, B01, B10, B11 (each 2x2)
    # C has blocks C00, C01, C10, C11 (each 2x2)
    
    # C00 = A00*B00 + A01*B10
    # C01 = A00*B01 + A01*B11
    # C10 = A10*B00 + A11*B10
    # C11 = A10*B01 + A11*B11
    
    # Each of these is a sum of two 2x2x2 tensor products.
    # So we can apply Strassen to each 2x2x2 product.
    
    # Total number of 2x2x2 products: 4 C-blocks * 2 products each = 8 products
    # Each product has rank 7, so total rank = 8 * 7 = 56
    
    # Let me redo this with rank = 56
    
    rank = 56
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # For each pair of blocks (block_a, block_b) that contribute to a C-block:
    # block_a is from A, block_b is from B
    # The column offset for each Strassen product is 7 * product_index
    
    # Product index: we have 8 products:
    # 0: A00 * B00 -> C00
    # 1: A01 * B10 -> C00
    # 2: A00 * B01 -> C01
    # 3: A01 * B11 -> C01
    # 4: A10 * B00 -> C10
    # 5: A11 * B10 -> C10
    # 6: A10 * B01 -> C11
    # 7: A11 * B11 -> C11
    
    # For product p, the block indices are:
    products = [
        (0, 0, 0),  # A block row=0,col=0; B block row=0,col=0; C block row=0,col=0
        (0, 1, 0),  # A block row=0,col=1; B block row=1,col=0; C block row=0,col=0
        (0, 0, 1),  # A block row=0,col=0; B block row=0,col=1; C block row=0,col=1
        (0, 1, 1),  # A block row=0,col=1; B block row=1,col=1; C block row=0,col=1
        (1, 0, 0),  # A block row=1,col=0; B block row=0,col=0; C block row=1,col=0
        (1, 1, 0),  # A block row=1,col=1; B block row=1,col=0; C block row=1,col=0
        (1, 0, 1),  # A block row=1,col=0; B block row=0,col=1; C block row=1,col=1
        (1, 1, 1),  # A block row=1,col=1; B block row=1,col=1; C block row=1,col=1
    ]
    
    for prod_idx, (ai, aj, bi, bj, ci, cj) in enumerate([
        (0, 0, 0, 0, 0, 0),
        (0, 1, 1, 0, 0, 0),
        (0, 0, 0, 1, 0, 1),
        (0, 1, 1, 1, 0, 1),
        (1, 0, 0, 0, 1, 0),
        (1, 1, 1, 0, 1, 0),
        (1, 0, 0, 1, 1, 1),
        (1, 1, 1, 1, 1, 1),
    ]):
        col_offset = prod_idx * 7
        for s in range(7):
            col = col_offset + s
            # U entries
            for local_a in range(4):
                if U2[local_a][s] != 0:
                    # Global row for A block (ai, aj), local index local_a
                    # local_a = 2*li + lj where li, lj in {0,1}
                    li = local_a // 2
                    lj = local_a % 2
                    global_row = 2 * ai + li
                    global_col_a = 2 * aj + lj
                    # U is indexed by row = global_row * m + global_col_a? No.
                    # U has shape (n*m, rank) = (16, rank)
                    # Row index corresponds to (i, j) in A, where i is row, j is col
                    # So row = i * m + j = i * 4 + j
                    U[global_row * m + global_col_a][col] = U2[local_a][s]
            
            # V entries
            for local_b in range(4):
                if V2[local_b][s] != 0:
                    li = local_b // 2
                    lj = local_b % 2
                    global_row = 2 * bi + li
                    global_col_b = 2 * bj + lj
                    # V is indexed by (j, k) in B, so row = j * p + k = j * 4 + k
                    V[global_row * p + global_col_b][col] = V2[local_b][s]
            
            # W entries
            for local_c in range(4):
                if W2[local_c][s] != 0:
                    li = local_c // 2
                    lj = local_c % 2
                    global_row = 2 * ci + li
                    global_col_c = 2 * cj + lj
                    # W is indexed by (k, i) in C following trace convention W[k*n+i, r]
                    # So row = k * n + i = k * 4 + i
                    # Here k is the column index in C, i is the row index in C
                    W[global_col_c * n + global_row][col] = W2[local_c][s]
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
