"""Matrix multiplication 6x6x6 - Block-Strassen decomposition, rank 189."""
DIMENSIONS = [6, 6, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 6, 6, 6
    # We decompose 6x6 as 3x3 blocks of size 2x2.
    # Each 2x2 block multiplication uses Strassen's 7-term algorithm.
    # Total rank = 3*3*7 = 189.
    
    # Strassen's 7 products for computing C = A*B where A, B, C are 2x2:
    # A = [[a11, a12], [a21, a22]], B = [[b11, b12], [b21, b22]]
    # p1 = (a11 + a22) * (b11 + b22)
    # p2 = (a21 + a22) * b11
    # p3 = a11 * (b12 - b22)
    # p4 = a22 * (b21 - b11)
    # p5 = (a11 + a12) * b22
    # p6 = (a21 - a11) * (b11 + b12)
    # p7 = (a12 - a22) * (b21 + b22)
    # C11 = p1 + p4 - p5 + p7
    # C12 = p3 + p5
    # C21 = p2 + p4
    # C22 = p1 - p2 + p3 + p6
    
    # For each Strassen term t (0..6), define:
    #   U_coeffs: dict mapping (row_idx_in_2x2) -> coeff  (row_idx 0,1 correspond to i in {0,1} within the 2x2 A block)
    #   V_coeffs: dict mapping (col_idx_in_2x2) -> coeff  (col_idx 0,1 correspond to k in {0,1} within the 2x2 B block)
    #   W_coeffs: dict mapping (out_idx_in_2x2) -> coeff  (out_idx 0,1 correspond to output positions within the 2x2 C block)
    
    # A indices: row r (0 or 1), which corresponds to element A[r][c] where c is the column within A's 2x2 block
    # In the tensor: U is indexed by (i_block*2 + r, j_block*2 + c) for A element
    # V is indexed by (j_block*2 + c, k_block*2 + d) for B element
    # W is indexed by (k_block*2 + d, i_block*2 + r) for C element (trace convention W[k*n+i, r])
    
    # Let me be more careful with the index conventions.
    # Matrix A is n x m = 6x6. Element A[i,j] where i=0..5, j=0..5.
    # Matrix B is m x p = 6x6. Element B[j,k] where j=0..5, k=0..5.
    # Matrix C is p x n = 6x6. Element C[k,i] where k=0..5, i=0..5.
    # 
    # U shape: (n*m, rank) = (36, rank). U[i*m + j, col] gives coefficient for A[i,j] in term col.
    # V shape: (m*p, rank) = (36, rank). V[j*p + k, col] gives coefficient for B[j,k] in term col.
    # W shape: (p*n, rank) = (36, rank). W[k*n + i, col] gives coefficient for C[k,i] in term col.
    # 
    # Block structure: 
    #   A is divided into 3x3 blocks of 2x2. Block (bi, bj) contains A[2*bi + r, 2*bj + c] for r,c in {0,1}.
    #   B is divided into 3x3 blocks of 2x2. Block (bj, bk) contains B[2*bj + c, 2*bk + d] for c,d in {0,1}.
    #   C is divided into 3x3 blocks of 2x2. Block (bk, bi) contains C[2*bk + d, 2*bi + r] for d,r in {0,1}.
    #
    # For block product C_block(bk,bi) = sum over bj of A_block(bi,bj) * B_block(bj,bk):
    #   Each such product uses 7 Strassen terms.
    #   Total terms: 3 * 3 * 3 * 7 = 189.
    #
    # For a given (bi, bj, bk, t) where t is the Strassen term index (0..6):
    #   U contributions: for each (r, c) in {0,1}x{0,1}, U[(2*bi+r)*6 + (2*bj+c)][col] += U_coeffs[t][r][c]
    #   V contributions: for each (c, d) in {0,1}x{0,1}, V[(2*bj+c)*6 + (2*bk+d)][col] += V_coeffs[t][c][d]
    #   W contributions: for each (d, r) in {0,1}x{0,1}, W[(2*bk+d)*6 + (2*bi+r)][col] += W_coeffs[t][d][r]

    # Define Strassen terms.
    # For A (2x2), elements are A[0][0], A[0][1], A[1][0], A[1][1] indexed as (r,c) with r,c in {0,1}.
    # For B (2x2), elements are B[0][0], B[0][1], B[1][0], B[1][1] indexed as (c,d) where c is row-in-A-block=col, d is col-in-B-block.
    # Actually let me re-derive. In block (bi, bj) of A, the 2x2 submatrix has entries:
    #   A[2*bi + 0][2*bj + 0], A[2*bi + 0][2*bj + 1]
    #   A[2*bi + 1][2*bj + 0], A[2*bi + 1][2*bj + 1]
    # Let's call these a00, a01, a10, a11 (using local indices (r,c)).
    #
    # In block (bj, bk) of B, the 2x2 submatrix has entries:
    #   B[2*bj + 0][2*bk + 0], B[2*bj + 0][2*bk + 1]
    #   B[2*bj + 1][2*bk + 0], B[2*bj + 1][2*bk + 1]
    # Let's call these b00, b01, b10, b11 (using local indices (c,d) where c is the shared index with A's column).
    #
    # The product C_block has entries:
    #   C[2*bk + 0][2*bi + 0] = a00*b00 + a01*b10
    #   C[2*bk + 0][2*bi + 1] = a00*b01 + a01*b11
    #   C[2*bk + 1][2*bi + 0] = a10*b00 + a11*b10
    #   C[2*bk + 1][2*bi + 1] = a10*b01 + a11*b11
    #
    # Strassen's algorithm:
    # p1 = (a00 + a11) * (b00 + b11)
    # p2 = (a10 + a11) * b00
    # p3 = a00 * (b01 - b11)
    # p4 = a11 * (b10 - b00)
    # p5 = (a00 + a01) * b11
    # p6 = (a10 - a00) * (b00 + b01)
    # p7 = (a01 - a11) * (b10 + b11)
    #
    # C00 = p1 + p4 - p5 + p7
    # C01 = p3 + p5
    # C10 = p2 + p4
    # C11 = p1 - p2 + p3 + p6

    # U_coeffs[t] is a 2x2 matrix of coefficients for A elements (r, c)
    # V_coeffs[t] is a 2x2 matrix of coefficients for B elements (c, d)
    # W_coeffs[t] is a 2x2 matrix of coefficients for C elements (d, r) [since C[k*n+i], k is row, i is col]
    # Wait, W is indexed by W[k*n + i, col]. In block (bk, bi), the C elements are C[2*bk+d][2*bi+r].
    # So W index is (2*bk+d)*6 + (2*bi+r). The local indices are (d, r).

    U_coeffs = [
        # p1 = (a00 + a11) * (b00 + b11)
        [[1, 0], [0, 1]],
        # p2 = (a10 + a11) * b00
        [[0, 0], [1, 1]],
        # p3 = a00 * (b01 - b11)
        [[1, 0], [0, 0]],
        # p4 = a11 * (b10 - b00)
        [[0, 0], [0, 1]],
        # p5 = (a00 + a01) * b11
        [[1, 1], [0, 0]],
        # p6 = (a10 - a00) * (b00 + b01)
        [[-1, 0], [1, 0]],
        # p7 = (a01 - a11) * (b10 + b11)
        [[0, 1], [0, -1]],
    ]
    
    V_coeffs = [
        # p1
        [[1, 0], [0, 1]],
        # p2
        [[1, 0], [0, 0]],
        # p3
        [[0, 1], [0, -1]],
        # p4
        [[-1, 0], [1, 0]],
        # p5
        [[0, 0], [0, 1]],
        # p6
        [[1, 1], [0, 0]],
        # p7
        [[0, 0], [1, 1]],
    ]
    
    W_coeffs = [
        # C00 = p1 + p4 - p5 + p7
        [[1, 0], [0, 0]],
        # C10 = p2 + p4
        [[0, 0], [1, 0]],
        # C01 = p3 + p5
        [[0, 0], [0, 0]],
        # C10 += p4, C11 += ... let me redo this.
        # W_coeffs[t][d][r] is the coefficient for C element at local position (d, r) in term t.
        # C00 (d=0,r=0): p1+1, p4+1, p5-1, p7+1
        # C01 (d=0,r=1): p3+1, p5+1
        # C10 (d=1,r=0): p2+1, p4+1
        # C11 (d=1,r=1): p1+1, p2-1, p3+1, p6+1
    ]
    
    # Let me redefine W_coeffs properly as a list of 7 terms, each a 2x2 matrix (d, r):
    W_coeffs = [
        # t=0 (p1): C00+=1, C11+=1
        [[1, 0], [0, 1]],
        # t=1 (p2): C10+=1, C11+=-1
        [[0, 0], [1, -1]],
        # t=2 (p3): C01+=1, C11+=1
        [[0, 1], [0, 1]],
        # t=3 (p4): C00+=1, C10+=1
        [[1, 0], [1, 0]],
        # t=4 (p5): C00+=-1, C01+=1
        [[-1, 1], [0, 0]],
        # t=5 (p6): C11+=1
        [[0, 0], [0, 1]],
        # t=6 (p7): C00+=1
        [[1, 0], [0, 0]],
    ]
    
    rank = 189
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    for bi in range(3):
        for bj in range(3):
            for bk in range(3):
                for t in range(7):
                    # Add U contributions
                    for r in range(2):
                        for c in range(2):
                            val = U_coeffs[t][r][c]
                            if val != 0:
                                U[(2*bi + r) * m + (2*bj + c)][col] += val
                    # Add V contributions
                    for c in range(2):
                        for d in range(2):
                            val = V_coeffs[t][c][d]
                            if val != 0:
                                V[(2*bj + c) * p + (2*bk + d)][col] += val
                    # Add W contributions
                    for d in range(2):
                        for r in range(2):
                            val = W_coeffs[t][d][r]
                            if val != 0:
                                W[(2*bk + d) * n + (2*bi + r)][col] += val
                    col += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
