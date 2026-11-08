"""Program for 4x4 matrix multiplication tensor decomposition using nested Strassen."""

def construct():
    n, m, p = 4, 4, 4
    
    # Nested Strassen decomposition
    # Outer level: 7 scalar products (Strassen for 2x2 block matrices of 2x2)
    # Inner level: 7 scalar products per block multiplication (Strassen for 2x2)
    # Total: 7 * 7 = 49 scalar products
    
    rank = 49
    U = [[0] * rank for _ in range(n * m)]  # (16, 49)
    V = [[0] * rank for _ in range(m * p)]  # (16, 49)
    W = [[0] * rank for _ in range(p * n)]  # (16, 49)
    
    # Strassen for 2x2:
    # M1 = (A00+A11)*(B00+B11)
    # M2 = (A10+A11)*B00
    # M3 = A00*(B10-B11)
    # M4 = A11*(B01-B10)
    # M5 = (A00+A01)*B11
    # M6 = (A10-A00)*(B00+B10)
    # M7 = (A01-A11)*(B01+B11)
    #
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # For each outer Strassen product (s=1..7), we compute block products
    # Each block product uses inner Strassen (t=1..7)
    
    # Outer Strassen coefficients for A-blocks:
    # s=1: A00 + A11
    # s=2: A10 + A11
    # s=3: A00
    # s=4: A11
    # s=5: A00 + A01
    # s=6: A10 - A00
    # s=7: A01 - A11
    
    # Outer Strassen coefficients for B-blocks:
    # s=1: B00 + B11
    # s=2: B00
    # s=3: B10 - B11
    # s=4: B01 - B10
    # s=5: B11
    # s=6: B00 + B10
    # s=7: B01 + B11
    
    # Block index mapping for 2x2 blocks of 2x2 matrices:
    # Block (0,0) -> rows 0-1, cols 0-1 in original 4x4
    # Block (0,1) -> rows 0-1, cols 2-3
    # Block (1,0) -> rows 2-3, cols 0-1
    # Block (1,1) -> rows 2-3, cols 2-3
    
    # U index: i*m + j = i*4 + j where i is row (0-3), j is col (0-3)
    # V index: j*p + k = j*4 + k where j is row (0-3), k is col (0-3)
    # W index: k*n + i = k*4 + i where k is row of B (0-3), i is col of C (0-3)
    
    # Inner Strassen for 2x2 blocks:
    # For a 2x2 block A_block with entries (a00, a01; a10, a11)
    # and B_block with entries (b00, b01; b10, b11)
    # The 7 products are:
    # t=1: (a00+a11)*(b00+b11)
    # t=2: (a10+a11)*b00
    # t=3: a00*(b10-b11)
    # t=4: a11*(b01-b10)
    # t=5: (a00+a01)*b11
    # t=6: (a10-a00)*(b00+b10)
    # t=7: (a01-a11)*(b01+b11)
    
    # Inner Strassen coefficients for A_block (2x2):
    # t=1: a00 + a11
    # t=2: a10 + a11
    # t=3: a00
    # t=4: a11
    # t=5: a00 + a01
    # t=6: a10 - a00
    # t=7: a01 - a11
    
    # Inner Strassen coefficients for B_block (2x2):
    # t=1: b00 + b11
    # t=2: b00
    # t=3: b10 - b11
    # t=4: b01 - b10
    # t=5: b11
    # t=6: b00 + b10
    # t=7: b01 + b11
    
    # Inner Strassen output coefficients for C_block (2x2):
    # t=1: c00 += 1, c11 += 1 (wait, let me re-derive)
    # Actually:
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # So for C_block entries:
    # c00: t=1:+1, t=4:+1, t=5:-1, t=7:+1
    # c01: t=3:+1, t=5:+1
    # c10: t=2:+1, t=4:+1
    # c11: t=1:+1, t=2:-1, t=3:+1, t=6:+1
    
    # Outer Strassen output coefficients for C (2x2 block matrix):
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # Now I need to compose: for each outer product s and inner product t,
    # determine which entries of U, V, W get which coefficients.
    
    # Block position mapping:
    # Block (bi, bj) for i-block and j-block
    # Rows in A: bi*2 to bi*2+1
    # Cols in A: bj*2 to bj*2+1
    # For U: row index = (bi*2 + ri)*4 + (bj*2 + cj) where ri, cj in {0,1}
    # Actually U has shape (n*m, rank) = (16, 49)
    # U[row, col] where row = i*4 + j, i is row of A, j is col of A
    
    # Let me define:
    # For outer block (bi, bj) where bi, bj in {0,1}:
    #   A entries: rows bi*2 to bi*2+1, cols bj*2 to bj*2+1
    #   U row index: (bi*2 + a_row)*4 + (bj*2 + a_col) where a_row, a_col in {0,1}
    
    # For inner Strassen within block (bi, bj):
    #   a00 -> U row: (bi*2+0)*4 + (bj*2+0)
    #   a01 -> U row: (bi*2+0)*4 + (bj*2+1)
    #   a10 -> U row: (bi*2+1)*4 + (bj*2+0)
    #   a11 -> U row: (bi*2+1)*4 + (bj*2+1)
    
    # Similarly for V (B matrix):
    #   Block (bk, bl) where bk, bl in {0,1}
    #   V row index: (bk*2 + b_row)*4 + (bl*2 + b_col) where b_row, b_col in {0,1}
    
    # For W (output C):
    #   C entry at (ci, cj) where ci is row, cj is col
    #   W index: cj*4 + ci (trace convention: W[k*n+i, r] where k is col of B = row of C, i is row of A = col of C)
    #   Wait, the task says: W[k*n+i, r]
    #   In matrix multiplication C = A*B, C[i,j] = sum_k A[i,k]*B[k,j]
    #   The tensor T[i,j,k] = A[i,k]*B[k,j]*C[j,i]... no.
    #   Standard: sum_k A[i,k] * B[k,j] = C[i,j]
    #   Tensor form: T[i,j,k] such that C[i,j] = sum_k A[i,k] B[k,j]
    #   W index convention: W[k*n+i, r] where this corresponds to the C[i,j] term?
    #   Let me re-read: "output factor follows the official trace convention W[k*n+i,r]"
    #   In the schoolbook code: W[k * n + i][column] = 1 where i is row of A, j is col of A = row of B, k is col of B
    #   So W index = k*n + i where i = row of A (0-3), k = col of B (0-3)
    #   This corresponds to C[i, k] in the product C = A*B
    #   So W[k*4 + i, r] contributes to C[i, k]
    
    # Now, for the composition:
    # Outer product s uses A-blocks and B-blocks
    # Let me define the outer block coefficients:
    
    # Outer A-block coefficients (which 2x2 blocks of A to add/subtract):
    # s=1: A00 + A11 -> blocks (0,0) and (1,1)
    # s=2: A10 + A11 -> blocks (1,0) and (1,1)
    # s=3: A00 -> block (0,0)
    # s=4: A11 -> block (1,1)
    # s=5: A00 + A01 -> blocks (0,0) and (0,1)
    # s=6: A10 - A00 -> blocks (1,0) and (0,0) with signs +1, -1
    # s=7: A01 - A11 -> blocks (0,1) and (1,1) with signs +1, -1
    
    # Outer B-block coefficients:
    # s=1: B00 + B11 -> blocks (0,0) and (1,1)
    # s=2: B00 -> block (0,0)
    # s=3: B10 - B11 -> blocks (1,0) and (1,1) with signs +1, -1
    # s=4: B01 - B10 -> blocks (0,1) and (1,0) with signs +1, -1
    # s=5: B11 -> block (1,1)
    # s=6: B00 + B10 -> blocks (0,0) and (1,0) with signs +1, +1
    # s=7: B01 + B11 -> blocks (0,1) and (1,1) with signs +1, +1
    
    # For each outer s, the "A-side" is a sum of A-blocks with coefficients
    # and the "B-side" is a sum of B-blocks with coefficients
    # The product is (sum of A-blocks) * (sum of B-blocks)
    # But wait, in Strassen, we precompute the sums, so each product is:
    # M_s = (sum_a alpha_a * A_block_a) * (sum_b beta_b * B_block_b)
    # This is not a single block multiplication, but we can't directly use inner Strassen
    # because the inputs to inner Strassen are already summed.
    
    # Hmm, this is the issue. In nested Strassen, we need to be careful.
    # Actually, the standard nested Strassen works as follows:
    # At the outer level, we form 7 linear combinations of 2x2 blocks.
    # Each such combination is a 2x2 matrix (sum of blocks).
    # Then we multiply these 2x2 matrices using... wait, no.
    # In Strassen for 2x2, we multiply 2x2 matrices. If our "scalars" are 2x2 blocks,
    # then we're doing matrix multiplication of 2x2 matrices, which requires 8 multiplications
    # unless we use Strassen again on the 2x2 matrix multiplication.
    
    # So for each outer product s:
    # X_s = sum of A-blocks (a 2x2 matrix)
    # Y_s = sum of B-blocks (a 2x2 matrix)
    # We need to compute X_s * Y_s where X_s and Y_s are 2x2 matrices.
    # We use inner Strassen for this 2x2 * 2x2 multiplication.
    # The inner Strassen gives 7 products, each of which is a product of linear combinations
    # of entries of X_s and Y_s.
    # Each entry of X_s is a linear combination of entries of A (from the block sums).
    # So each inner product becomes a product of two linear combinations of A-entries and B-entries.
    
    # Let me just enumerate all 49 products explicitly.
    
    # For outer s and inner t, the product is:
    # P(s,t) = (linear combo of A entries) * (linear combo of B entries)
    
    # A-side for outer s:
    # s=1: A00 + A11 -> entries: A[0,0], A[0,1], A[1,0], A[1,1], A[2,2], A[2,3], A[3,2], A[3,3]
    #       with all coefficients +1
    # s=2: A10 + A11 -> entries: A[2,0], A[2,1], A[3,0], A[3,1], A[2,2], A[2,3], A[3,2], A[3,3]
    #       with all coefficients +1
    # s=3: A00 -> entries: A[0,0], A[0,1], A[1,0], A[1,1] with +1
    # s=4: A11 -> entries: A[2,2], A[2,3], A[3,2], A[3,3] with +1
    # s=5: A00 + A01 -> entries: A[0,0], A[0,1], A[1,0], A[1,1], A[0,2], A[0,3], A[1,2], A[1,3]
    #       with all +1
    # s=6: A10 - A00 -> entries: A[2,0], A[2,1], A[3,0], A[3,1] with +1; A[0,0], A[0,1], A[1,0], A[1,1] with -1
    # s=7: A01 - A11 -> entries: A[0,2], A[0,3], A[1,2], A[1,3] with +1; A[2,2], A[2,3], A[3,2], A[3,3] with -1
    
    # Let me represent A-side coefficients as a dict from (i,j) to coeff for each s.
    
    def get_a_coeff(s, i, j):
        # Returns coefficient of A[i,j] in the A-side of outer product s
        # Block (bi, bj) contains A[bi*2+ri, bj*2+cj] for ri,cj in {0,1}
        bi = i // 2
        bj = j // 2
        # Determine which outer products include block (bi,bj) with what coefficient
        coeff = 0
        if s == 1:
            if (bi, bj) in [(0,0), (1,1)]:
                coeff = 1
        elif s == 2:
            if (bi, bj) in [(1,0), (1,1)]:
                coeff = 1
        elif s == 3:
            if (bi, bj) == (0,0):
                coeff = 1
        elif s == 4:
            if (bi, bj) == (1,1):
                coeff = 1
        elif s == 5:
            if (bi, bj) in [(0,0), (0,1)]:
                coeff = 1
        elif s == 6:
            if (bi, bj) == (1,0):
                coeff = 1
            elif (bi, bj) == (0,0):
                coeff = -1
        elif s == 7:
            if (bi, bj) == (0,1):
                coeff = 1
            elif (bi, bj) == (1,1):
                coeff = -1
        return coeff
    
    def get_b_coeff(s, i, j):
        # Returns coefficient of B[i,j] in the B-side of outer product s
        bi = i // 2  # row block of B
        bj = j // 2  # col block of B
        coeff = 0
        if s == 1:
            if (bi, bj) in [(0,0), (1,1)]:
                coeff = 1
        elif s == 2:
            if (bi, bj) == (0,0):
                coeff = 1
        elif s == 3:
            if (bi, bj) == (1,0):
                coeff = 1
            elif (bi, bj) == (1,1):
                coeff = -1
        elif s == 4:
            if (bi, bj) == (0,1):
                coeff = 1
            elif (bi, bj) == (1,0):
                coeff = -1
        elif s == 5:
            if (bi, bj) == (1,1):
                coeff = 1
        elif s == 6:
            if (bi, bj) in [(0,0), (1,0)]:
                coeff = 1
        elif s == 7:
            if (bi, bj) in [(0,1), (1,1)]:
                coeff = 1
        return coeff
    
    # Inner Strassen coefficients for 2x2 matrix multiplication:
    # X * Y where X, Y are 2x2
    # t=1: (X00+X11)*(Y00+Y11)
    # t=2: (X10+X11)*Y00
    # t=3: X00*(Y10-Y11)
    # t=4: X11*(Y01-Y10)
    # t=5: (X00+X01)*Y11
    # t=6: (X10-X00)*(Y00+Y10)
    # t=7: (X01-X11)*(Y01+Y11)
    
    def get_inner_x_coeff(t, x_row, x_col):
        # Coefficient of X[x_row, x_col] in the X-side of inner product t
        coeff = 0
        if t == 1:
            if (x_row, x_col) in [(0,0), (1,1)]:
                coeff = 1
        elif t == 2:
            if (x_row, x_col) in [(1,0), (1,1)]:
                coeff = 1
        elif t == 3:
            if (x_row, x_col) == (0,0):
                coeff = 1
        elif t == 4:
            if (x_row, x_col) == (1,1):
                coeff = 1
        elif t == 5:
            if (x_row, x_col) in [(0,0), (0,1)]:
                coeff = 1
        elif t == 6:
            if (x_row, x_col) == (1,0):
                coeff = 1
            elif (x_row, x_col) == (0,0):
                coeff = -1
        elif t == 7:
            if (x_row, x_col) == (0,1):
                coeff = 1
            elif (x_row, x_col) == (1,1):
                coeff = -1
        return coeff
    
    def get_inner_y_coeff(t, y_row, y_col):
        # Coefficient of Y[y_row, y_col] in the Y-side of inner product t
        coeff = 0
        if t == 1:
            if (y_row, y_col) in [(0,0), (1,1)]:
                coeff = 1
        elif t == 2:
            if (y_row, y_col) == (0,0):
                coeff = 1
        elif t == 3:
            if (y_row, y_col) == (1,0):
                coeff = 1
            elif (y_row, y_col) == (1,1):
                coeff = -1
        elif t == 4:
            if (y_row, y_col) == (0,1):
                coeff = 1
            elif (y_row, y_col) == (1,0):
                coeff = -1
        elif t == 5:
            if (y_row, y_col) == (1,1):
                coeff = 1
        elif t == 6:
            if (y_row, y_col) in [(0,0), (1,0)]:
                coeff = 1
        elif t == 7:
            if (y_row, y_col) in [(0,1), (1,1)]:
                coeff = 1
        return coeff
    
    # Inner Strassen output: C_block = X * Y
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    def get_inner_c_coeff(t, c_row, c_col):
        # Coefficient of product t in C_block[c_row, c_col]
        coeff = 0
        if (c_row, c_col) == (0,0):
            if t in [1, 4, 7]:
                coeff = 1
            elif t == 5:
                coeff = -1
        elif (c_row, c_col) == (0,1):
            if t in [3, 5]:
                coeff = 1
        elif (c_row, c_col) == (1,0):
            if t in [2, 4]:
                coeff = 1
        elif (c_row, c_col) == (1,1):
            if t in [1, 3, 6]:
                coeff = 1
            elif t == 2:
                coeff = -1
        return coeff
    
    # Outer Strassen output: C_block_matrix
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    def get_outer_c_coeff(s, c_block_row, c_block_col):
        # Coefficient of product s in C_block_matrix[c_block_row, c_block_col]
        coeff = 0
        if (c_block_row, c_block_col) == (0,0):
            if s in [1, 4, 7]:
                coeff = 1
            elif s == 5:
                coeff = -1
        elif (c_block_row, c_block_col) == (0,1):
            if s in [3, 5]:
                coeff = 1
        elif (c_block_row, c_block_col) == (1,0):
            if s in [2, 4]:
                coeff = 1
        elif (c_block_row, c_block_col) == (1,1):
            if s in [1, 3, 6]:
                coeff = 1
            elif s == 2:
                coeff = -1
        return coeff
    
    # Now build U, V, W
    # For each outer s and inner t, we have a product:
    # P(s,t) = (sum over A entries: a_coeff(s, i, j) * get_inner_x_coeff(t, xi, xj) * A[i,j])
    #          * (sum over B entries: b_coeff(s, k, l) * get_inner_y_coeff(t, yi, yj) * B[k,l])
    # where xi, xj are the positions within the block for the X-side,
    # and yi, yj are the positions within the block for the Y-side.
    
    # Actually, I need to be more careful. The X-side of inner Strassen operates on the 2x2 matrix X_s
    # which is the A-side of outer product s. X_s is a 2x2 matrix where:
    # X_s[ri, cj] = sum over blocks (bi, bj) of a_coeff(s, bi, bj) * A[bi*2+ri, bj*2+cj]
    # Wait, no. X_s is formed by summing A-blocks. Each A-block is 2x2.
    # X_s[ri, cj] = sum_{bi,bj} a_block_coeff(s, bi, bj) * A[bi*2+ri, bj*2+cj]
    # where a_block_coeff is the coefficient of block (bi,bj) in the outer sum.
    
    # Let me redefine: for outer product s, the A-side is a 2x2 matrix X_s where
    # X_s[ri, cj] = sum over (bi, bj) of outer_a_block_coeff(s, bi, bj) * A[bi*2+ri, bj*2+cj]
    
    # Similarly, B-side Y_s[kj, cj] = sum over (bk, bl) of outer_b_block_coeff(s, bk, bl) * B[bk*2+kj, bl*2+cj]
    
    # Then inner Strassen computes X_s * Y_s using 7 products.
    # The result is a 2x2 matrix Z_s = X_s * Y_s.
    # Z_s[ri, cj] = sum over (bi, bj, bk, bl) of ... this gets complicated.
    
    # Let me just directly compute the coefficients for each A entry and B entry for each (s, t) product.
    
    # For product (s, t):
    # U-side: coefficient of A[i, j] = outer_a_coeff(s, i, j) * inner_x_coeff(t, i%2, j%2)
    # Wait, that's not right either. Let me think again.
    
    # X_s[ri, cj] = sum_{bi,bj} outer_a_block_coeff(s, bi, bj) * A[bi*2+ri, bj*2+cj]
    # So for a given A entry A[i, j], where i = bi*2+ri, j = bj*2+cj:
    # It contributes to X_s[ri, cj] with coefficient outer_a_block_coeff(s, bi, bj)
    
    # Inner Strassen X-side for product t:
    # X_t[ri, cj] = sum over positions (pr, pc) of inner_x_coeff(t, pr, pc) * X_s[pr, pc]
    
    # So A[i, j] contributes to X_t[ri, cj] with coefficient:
    # sum over (pr, pc) of inner_x_coeff(t, pr, pc) * [if (ri, cj) == (pr, pc) then outer_a_block_coeff(s, bi, bj) else 0]
    # = inner_x_coeff(t, ri, cj) * outer_a_block_coeff(s, bi, cj)
    
    # Wait, I'm confusing myself. Let me be very explicit.
    
    # A[i, j] where i = 2*bi + ri, j = 2*bj + cj, ri, cj in {0,1}, bi, bj in {0,1}
    # This entry is in block (bi, bj) of A.
    # In X_s, the entry at position (ri, cj) gets contribution outer_a_block_coeff(s, bi, bj) * A[i, j]
    # So A[i, j] contributes to X_s[ri, cj] with coefficient outer_a_block_coeff(s, bi, bj)
    
    # For inner Strassen product t, the X-side linear combination is:
    # sum_{pr, pc in {0,1}} inner_x_coeff(t, pr, pc) * X_s[pr, pc]
    # A[i, j] contributes to this sum as:
    # sum_{pr, pc} inner_x_coeff(t, pr, pc) * [coefficient of A[i,j] in X_s[pr, pc]]
    # = sum_{pr, pc} inner_x_coeff(t, pr, pc) * [outer_a_block_coeff(s, bi, bj) if (pr, pc) == (ri, cj) else 0]
    # = inner_x_coeff(t, ri, cj) * outer_a_block_coeff(s, bi, bj)
    
    # So the total coefficient of A[i, j] in the X-side of product (s, t) is:
    # outer_a_block_coeff(s, bi, bj) * inner_x_coeff(t, ri, cj)
    # where bi = i//2, bj = j//2, ri = i%2, cj = j%2
    
    # Similarly for B[k, l] where k = 2*bk + rk, l = 2*bl + cl:
    # Coefficient in Y-side of product (s, t) is:
    # outer_b_block_coeff(s, bk, bl) * inner_y_coeff(t, rk, cl)
    
    # And for the output:
    # The product (s, t) contributes to Z_s[ri2, cj2] with coefficient inner_c_coeff(t, ri2, cj2)
    # where Z_s = X_s * Y_s
    # Then Z_s contributes to the final C with outer_c_coeff(s, bi_out, bj_out) for each block position
    # C entry at position (2*bi_out + ri2, 2*bj_out + cj2) gets contribution:
    # inner_c_coeff(t, ri2, cj2) * outer_c_coeff(s, bi_out, bj_out) * P(s,t)
    
    # So for W, the coefficient of C[i_out, j_out] in product (s, t) is:
    # Let i_out = 2*bi_out + ri2, j_out = 2*bj_out + cj2
    # W index = j_out * n + i_out = j_out * 4 + i_out
    # Coefficient = inner_c_coeff(t, ri2, cj2) * outer_c_coeff(s, bi_out, bj_out)
    
    # Let's code this up.
    
    for s in range(1, 8):
        for t in range(1, 8):
            col = (s-1)*7 + (t-1)  # 0-48
            
            # U: coefficient of A[i, j]
            for i in range(4):
                for j in range(4):
                    bi = i // 2
                    bj = j // 2
                    ri = i % 2
                    cj = j % 2
                    a_coeff = get_a_coeff(s, i, j)  # This gives the block coefficient
                    # Actually, get_a_coeff returns the coefficient based on block (bi, bj)
                    # Let me use a separate function for block coefficients
                    # I'll redefine: outer_a_block_coeff(s, bi, bj)
                    # and outer_b_block_coeff(s, bk, bl)
                    pass
            
            # Let me redefine the block coefficient functions properly
            # I'll just compute everything in nested loops.
    
    # Redefining with block coefficient functions:
    def outer_a_block_coeff(s, bi, bj):
        if s == 1:
            if (bi, bj) in [(0,0), (1,1)]:
                return 1
            return 0
        elif s == 2:
            if (bi, bj) in [(1,0), (1,1)]:
                return 1
            return 0
        elif s == 3:
            if (bi, bj) == (0,0):
                return 1
            return 0
        elif s == 4:
            if (bi, bj) == (1,1):
                return 1
            return 0
        elif s == 5:
            if (bi, bj) in [(0,0), (0,1)]:
                return 1
            return 0
        elif s == 6:
            if (bi, bj) == (1,0):
                return 1
            elif (bi, bj) == (0,0):
                return -1
            return 0
        elif s == 7:
            if (bi, bj) == (0,1):
                return 1
            elif (bi, bj) == (1,1):
                return -1
            return 0
        return 0
    
    def outer_b_block_coeff(s, bk, bl):
        if s == 1:
            if (bk, bl) in [(0,0), (1,1)]:
                return 1
            return 0
        elif s == 2:
            if (bk, bl) == (0,0):
                return 1
            return 0
        elif s == 3:
            if (bk, bl) == (1,0):
                return 1
            elif (bk, bl) == (1,1):
                return -1
            return 0
        elif s == 4:
            if (bk, bl) == (0,1):
                return 1
            elif (bk, bl) == (1,0):
                return -1
            return 0
        elif s == 5:
            if (bk, bl) == (1,1):
                return 1
            return 0
        elif s == 6:
            if (bk, bl) in [(0,0), (1,0)]:
                return 1
            return 0
        elif s == 7:
            if (bk, bl) in [(0,1), (1,1)]:
                return 1
            return 0
        return 0
    
    # Now build the matrices
    U = [[0] * rank for _ in range(16)]
    V = [[0] * rank for _ in range(16)]
    W = [[0] * rank for _ in range(16)]
    
    for s in range(1, 8):
        for t in range(1, 8):
            col = (s-1)*7 + (t-1)
            
            # U: A[i, j]
            for i in range(4):
                for j in range(4):
                    bi = i // 2
                    bj = j // 2
                    ri = i % 2
                    cj = j % 2
                    a_coeff = outer_a_block_coeff(s, bi, bj) * get_inner_x_coeff(t, ri, cj)
                    if a_coeff != 0:
                        U[i*4 + j][col] += a_coeff
            
            # V: B[k, l]
            for k in range(4):
                for l in range(4):
                    bk = k // 2
                    bl = l // 2
                    rk = k % 2
                    cl = l % 2
                    b_coeff = outer_b_block_coeff(s, bk, bl) * get_inner_y_coeff(t, rk, cl)
                    if b_coeff != 0:
                        V[k*4 + l][col] += b_coeff
            
            # W: C[i_out, j_out], W index = j_out*4 + i_out
            for i_out in range(4):
                for j_out in range(4):
                    bi_out = i_out // 2
                    bj_out = j_out // 2
                    ri2 = i_out % 2
                    cj2 = j_out % 2
                    c_coeff = outer_c_coeff(s, bi_out, bj_out) * get_inner_c_coeff(t, ri2, cj2)
                    if c_coeff != 0:
                        W[j_out*4 + i_out][col] += c_coeff
    
    return {"U": U, "V": V, "W": W}
