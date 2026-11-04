"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = 4, 4, 6

    # Rank-7 Strassen decomposition for 2x2x2 matrix multiplication.
    # A = [[a00, a01], [a10, a11]], B = [[b00, b01], [b10, b11]]
    # C = A*B where C[i,j] = sum_k A[i,k]B[k,j]
    #
    # The 7 multiplications are:
    # M1 = (A00 + A11)(B00 + B11)
    # M2 = (A10 + A11) B00
    # M3 = A00 (B10 - B00)
    # M4 = A11 (B01 - B11)
    # M5 = (A00 + A01) B11
    # M6 = (A10 - A00)(B00 + B01)
    # M7 = (A01 - A11)(B10 + B11)
    #
    # C00 = M1 - M5 + M4 + M7
    # C10 = M2 + M6
    # C01 = M3 + M5
    # C11 = M1 - M2 + M3 + M6

    # U vectors (for A): each is a 4-vector [a00, a01, a10, a11]
    U_strassen = [
        [1, 0, 0, 1],   # M1: A00 + A11
        [0, 0, 1, 1],   # M2: A10 + A11
        [1, 0, 0, 0],   # M3: A00
        [0, 0, 0, 1],   # M4: A11
        [1, 1, 0, 0],   # M5: A00 + A01
        [1, 0, -1, 0],  # M6: A10 - A00
        [0, 1, 0, -1],  # M7: A01 - A11
    ]

    # V vectors (for B): each is a 4-vector [b00, b01, b10, b11]
    V_strassen = [
        [1, 0, 0, 1],   # M1: B00 + B11
        [1, 0, 0, 0],   # M2: B00
        [0, 0, 1, -1],  # M3: B10 - B00
        [0, 1, 0, -1],  # M4: B01 - B11
        [0, 0, 0, 1],   # M5: B11
        [1, 1, 0, 0],   # M6: B00 + B01
        [0, 0, 1, 1],   # M7: B10 + B11
    ]

    # W vectors (for C): each is a 4-vector [c00, c10, c01, c11] in trace convention
    # Trace convention: W[k*n+i, r] where output index is k*n + i
    # For 2x2 block: positions are (0,0), (0,1), (1,0), (1,1)
    # In flattened form with n=2: index = col*2 + row
    # So: c00 -> 0, c01 -> 1, c10 -> 2, c11 -> 3
    # But the task says W[k*n+i, r] where the output index is k*n + i
    # For C[i,j], the index is j*n + i (column-major) or i*n + j (row-major)?
    # The task says "W[k*n+i,r]" which suggests column-major: index = k*n + i where k is column, i is row.
    # For 2x2: c00 -> 0*2+0=0, c10 -> 0*2+1=1, c01 -> 1*2+0=2, c11 -> 1*2+1=3
    # So W vector order is: [c00, c10, c01, c11]
    
    W_strassen = [
        [1, 0, 0, 1],   # M1: C00 += 1, C11 += 1
        [0, 1, 0, -1],  # M2: C10 += 1, C11 -= 1
        [0, 0, 1, 1],   # M3: C01 += 1, C11 += 1
        [1, 0, 0, 0],   # M4: C00 += 1
        [-1, 0, 1, 0],  # M5: C00 -= 1, C01 += 1
        [0, 1, 0, 1],   # M6: C10 += 1, C11 += 1
        [1, 0, 0, 0],   # M7: C00 += 1
    ]

    # Now construct the 4x4x6 decomposition.
    # A is 4x4, B is 4x6, C is 4x6.
    # We split into 2x2 blocks:
    # A has 2x2 blocks: A[i,j] for i,j in {0,1}, each 2x2
    # B has 2x2 blocks: B[j,k] for j in {0,1}, k in {0,1,2}, each 2x2
    # C has 2x2 blocks: C[i,k] for i in {0,1}, k in {0,1,2}, each 2x2
    #
    # C[i,k] = sum_{j=0}^{1} A[i,j] * B[j,k]
    #
    # This is a 2x2 matrix multiplication over the ring of 2x2 matrices.
    # We can use Strassen for the outer 2x2 multiplication.
    #
    # Outer Strassen:
    # X1 = A[0,0] + A[1,1]
    # X2 = A[1,0] + A[1,1]
    # X3 = A[0,0]
    # X4 = A[1,1]
    # X5 = A[0,0] + A[0,1]
    # X6 = A[1,0] - A[0,0]
    # X7 = A[0,1] - A[1,1]
    #
    # Y1 = B[0,0] + B[1,2]
    # Y2 = B[0,0]
    # Y3 = B[1,0] - B[0,0]
    # Y4 = B[1,1] - B[1,2]
    # Y5 = B[1,2]
    # Y6 = B[0,0] + B[0,1]
    # Y7 = B[1,0] + B[1,2]
    #
    # Z1 = X1 * Y1
    # Z2 = X2 * Y2
    # Z3 = X3 * Y3
    # Z4 = X4 * Y4
    # Z5 = X5 * Y5
    # Z6 = X6 * Y6
    # Z7 = X7 * Y7
    #
    # C[0,0] = Z1 - Z5 + Z4 + Z7
    # C[1,0] = Z2 + Z6
    # C[0,1] = Z3 + Z5
    # C[1,1] = Z1 - Z2 + Z3 + Z6
    # C[0,2] = Z5
    # C[1,2] = Z7
    #
    # Wait, that's not right. Let me re-derive.
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # C[0,0] = A[0,0]*B[0,0] + A[0,1]*B[1,0]
    # C[0,1] = A[0,0]*B[0,1] + A[0,1]*B[1,1]
    # C[0,2] = A[0,0]*B[0,2] + A[0,1]*B[1,2]
    # C[1,0] = A[1,0]*B[0,0] + A[1,1]*B[1,0]
    # C[1,1] = A[1,0]*B[0,1] + A[1,1]*B[1,1]
    # C[1,2] = A[1,0]*B[0,2] + A[1,1]*B[1,2]
    #
    # This is a 2x3 block matrix multiplication.
    # We can use Strassen for the 2x2 part and handle the third column separately.
    #
    # Let's use the fact that we can decompose this as:
    # 6 independent 2x2x2 multiplications (one for each (i,k) pair, summing over j)
    # Each has rank 7, so total rank = 6*7 = 42.
    
    # For each (i, k) where i in {0,1}, k in {0,1,2}:
    # C[i,k] = A[i,0]*B[0,k] + A[i,1]*B[1,k]
    # This is a 2x2 matrix multiplication with 2 terms.
    # We can use Strassen for each 2x2x2 multiplication.
    
    # Total rank = 6 * 7 = 42
    
    rank = 42
    U = [[0] * rank for _ in range(n * m)]  # 16 x 42
    V = [[0] * rank for _ in range(m * p)]  # 24 x 42
    W = [[0] * rank for _ in range(p * n)]  # 24 x 42
    
    col = 0
    for i_block in range(2):
        for k_block in range(3):
            # For this (i_block, k_block), we have a 2x2x2 multiplication
            # C[i_block, k_block] = A[i_block, 0] * B[0, k_block] + A[i_block, 1] * B[1, k_block]
            # This is a 2x2 matrix multiplication where:
            # A_local = [[A[i_block,0][0,0], A[i_block,0][0,1], A[i_block,1][0,0], A[i_block,1][0,1]],
            #            [A[i_block,0][1,0], A[i_block,0][1,1], A[i_block,1][1,0], A[i_block,1][1,1]]]
            # Wait, that's not right. Let me think again.
            #
            # C[i_block, k_block] is a 2x2 matrix.
            # C[i_block, k_block][r, c] = sum_{j=0}^{1} A[i_block, j][r, :] * B[j, k_block][:, c]
            # = A[i_block, 0][r, 0]*B[0, k_block][0, c] + A[i_block, 0][r, 1]*B[1, k_block][1, c] + ...
            #
            # This is a 2x2 matrix multiplication where the "scalars" are the entries.
            # We can use Strassen for each (r, c) pair, but that would be 4 independent 1x1x2 multiplications.
            #
            # Actually, let's just do the schoolbook for each 2x2x2 block multiplication.
            # For each (i_block, k_block), we have:
            # C[i_block, k_block][r, c] = sum_{j=0}^{1} sum_{s=0}^{1} A[i_block, j][r, s] * B[j, k_block][s, c]
            #
            # This is a 2x2x2 tensor multiplication with 2x2x2 = 8 terms.
            # We can use Strassen to reduce this to 7 terms.
            
            # Let's use the Strassen decomposition for each 2x2x2 block.
            # A_local is a 2x2 matrix: A_local = [[A[i_block,0][r,0], A[i_block,0][r,1]], [A[i_block,1][r,0], A[i_block,1][r,1]]]
            # Wait, that's not right either.
            
            # Let me just use the schoolbook for now to get a correct decomposition.
            # Rank = 4*4*6 = 96 for schoolbook.
            # But we can do better with Strassen.
            
            # For each (i_block, k_block), the multiplication is:
            # C[i_block, k_block] = A[i_block, 0] * B[0, k_block] + A[i_block, 1] * B[1, k_block]
            # where A[i_block, j] and B[j, k_block] are 2x2 matrices.
            #
            # This is equivalent to a 2x2 matrix multiplication over the ring of 2x2 matrices.
            # The rank of this is at most 7 (using Strassen for the outer 2x2) * 4 (for the inner 2x2) = 28?
            # No, that's not right.
            
            # Let's just use the schoolbook for each entry.
            # C[i_block, k_block][r, c] = sum_{j=0}^{1} sum_{s=0}^{1} A[i_block, j][r, s] * B[j, k_block][s, c]
            # For each (i_block, k_block, r, c), we have 4 terms.
            # Total terms = 2 * 3 * 2 * 2 * 4 = 96.
            
            for r in range(2):
                for c in range(2):
                    for j in range(2):
                        for s in range(2):
                            # A index: (i_block*2 + r, j*2 + s)
                            a_idx = (i_block * 2 + r) * 4 + (j * 2 + s)
                            # B index: (j*2 + s, k_block*2 + c)
                            b_idx = (j * 2 + s) * 6 + (k_block * 2 + c)
                            # C index: (k_block*2 + c)*4 + (i_block*2 + r)  [trace convention: k*n+i]
                            c_idx = (k_block * 2 + c) * 4 + (i_block * 2 + r)
                            
                            U[a_idx][col] = 1
                            V[b_idx][col] = 1
                            W[c_idx][col] = 1
                            col += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
