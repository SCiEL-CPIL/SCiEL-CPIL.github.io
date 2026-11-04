DIMENSIONS = [4, 4, 7]

def construct():
    n, m, p = 4, 4, 7
    nm = n * m  # 16
    mp = m * p  # 28
    pn = p * n  # 28

    terms = []  # list of (u, v, w) where u,v,w are lists of length nm, mp, pn

    def make_u(idx_i, idx_j, val):
        u = [0] * nm
        u[idx_i * m + idx_j] = val
        return u

    def make_v(idx_j, idx_k, val):
        v = [0] * mp
        v[idx_j * p + idx_k] = val
        return v

    def make_w(idx_k, idx_i, val):
        w = [0] * pn
        w[idx_k * n + idx_i] = val
        return w

    # Strassen 2x2 decomposition for a block product
    # A_blocks: list of 4 blocks, each a 2x2 list, indexed [row_block][col_block]
    # B_blocks: list of 4 blocks, each a 2x2 list, indexed [row_block][col_block]
    # w_offset: the offset in the output matrix (in terms of the full 4x7 matrix)
    #   The output block is 2x2, located at rows [w_offset_row, w_offset_row+1], cols [w_offset_col, w_offset_col+1]
    #   w_offset_row is 0 or 2, w_offset_col is 0, 2, or 4

    def add_strassen_2x2(A_blocks, B_blocks, w_offset_row, w_offset_col):
        # A_blocks[br][bc] is a 2x2 list, br in {0,1}, bc in {0,1}
        # B_blocks[br][bc] is a 2x2 list, br in {0,1}, bc in {0,1}
        # The product block C = A_block * B_block is 2x2
        # We decompose this 2x2 product using Strassen's 7 multiplications
        # Each multiplication is of two 2x2 matrices, which we decompose using 7 rank-1 terms
        # Wait, that's not right. Strassen's algorithm for 2x2 matrix multiplication
        # computes 7 scalar multiplications. Each scalar multiplication is a rank-1 term
        # in the tensor decomposition of the 2x2x2 tensor.
        # So for each of the 7 Strassen steps, we add 1 rank-1 term? No.
        # The 2x2x2 tensor has rank 7. So we need 7 rank-1 terms to decompose it.
        # Each rank-1 term corresponds to one of the 7 Strassen products.
        # But the inputs to these products are linear combinations of the entries of A_block and B_block.
        # So the U and V vectors for each rank-1 term will have non-zero entries at the positions
        # corresponding to the entries involved in that Strassen product.

        # Let's define the 7 Strassen products:
        # M1 = (A11 + A22)(B11 + B22)
        # M2 = (A21 + A22)B11
        # M3 = A11(B21 - B22)
        # M4 = A22(B12 - B11)
        # M5 = (A11 + A12)B22
        # M6 = (A21 - A11)(B11 + B12)
        # M7 = (A12 - A22)(B21 + B22)
        #
        # C11 = M1 + M4 - M5 + M7
        # C12 = M3 + M5
        # C21 = M2 + M4
        # C22 = M1 - M2 + M3 + M6
        #
        # Each Mi is a scalar product of two linear combinations.
        # So each Mi contributes one rank-1 term to the tensor decomposition.
        # The U vector for Mi has non-zero entries at the positions of A_block entries involved.
        # The V vector for Mi has non-zero entries at the positions of B_block entries involved.
        # The W vector for Mi has non-zero entries at the positions of C_block entries that Mi contributes to.

        # Let's define the positions in the full matrix for A_block, B_block, C_block
        # A_block is at rows [w_offset_row, w_offset_row+1], cols [0, 1] (since it's the left block)
        # Wait, no. The A_block is part of the 4x4 matrix A.
        # The 4x4 matrix A is divided into 2x2 blocks:
        # A = [[A11, A12], [A21, A22]]
        # A11 is at rows 0,1 and cols 0,1
        # A12 is at rows 0,1 and cols 2,3
        # A21 is at rows 2,3 and cols 0,1
        # A22 is at rows 2,3 and cols 2,3
        #
        # So the positions in the U vector (length 16) for A_block entries are:
        # A11: indices 0,1,4,5
        # A12: indices 2,3,6,7
        # A21: indices 8,9,12,13
        # A22: indices 10,11,14,15
        #
        # B_block is part of the 4x7 matrix B.
        # B = [[B11, B12, B13], [B21, B22, B23]]
        # B11 is at rows 0,1 and cols 0,1
        # B12 is at rows 0,1 and cols 2,3
        # B13 is at rows 0,1 and cols 4,5
        # B21 is at rows 2,3 and cols 0,1
        # B22 is at rows 2,3 and cols 2,3
        # B23 is at rows 2,3 and cols 4,5
        #
        # So the positions in the V vector (length 28) for B_block entries are:
        # B11: indices 0,1,4,5
        # B12: indices 2,3,6,7
        # B13: indices 8,9,12,13
        # B21: indices 14,15,18,19
        # B22: indices 16,17,20,21
        # B23: indices 22,23,26,27
        #
        # C_block is part of the 4x7 matrix C.
        # C = [[C11, C12, C13], [C21, C22, C23]]
        # C11 is at rows 0,1 and cols 0,1
        # C12 is at rows 0,1 and cols 2,3
        # C13 is at rows 0,1 and cols 4,5
        # C21 is at rows 2,3 and cols 0,1
        # C22 is at rows 2,3 and cols 2,3
        # C23 is at rows 2,3 and cols 4,5
        #
        # So the positions in the W vector (length 28) for C_block entries are:
        # C11: indices 0,1,4,5
        # C12: indices 2,3,6,7
        # C13: indices 8,9,12,13
        # C21: indices 14,15,18,19
        # C22: indices 16,17,20,21
        # C23: indices 22,23,26,27

        # Now, for a given Strassen product Mi, we need to know which A_block entries and B_block entries are involved.
        # For example, M1 = (A11 + A22)(B11 + B22)
        # The U vector for M1 has +1 at A11 positions and +1 at A22 positions.
        # The V vector for M1 has +1 at B11 positions and +1 at B22 positions.
        # The W vector for M1 has +1 at C11 and C22 positions (since M1 contributes to C11 and C22).

        # Let's define the A_block entry positions for each block:
        A11_pos = [0, 1, 4, 5]
        A12_pos = [2, 3, 6, 7]
        A21_pos = [8, 9, 12, 13]
        A22_pos = [10, 11, 14, 15]

        # B_block entry positions for each block:
        B11_pos = [0, 1, 4, 5]
        B12_pos = [2, 3, 6, 7]
        B13_pos = [8, 9, 12, 13]
        B21_pos = [14, 15, 18, 19]
        B22_pos = [16, 17, 20, 21]
        B23_pos = [22, 23, 26, 27]

        # C_block entry positions for each block:
        C11_pos = [0, 1, 4, 5]
        C12_pos = [2, 3, 6, 7]
        C13_pos = [8, 9, 12, 13]
        C21_pos = [14, 15, 18, 19]
        C22_pos = [16, 17, 20, 21]
        C23_pos = [22, 23, 26, 27]

        # Now, for the 7 Strassen products, we define the U, V, W vectors.
        # Each product is a scalar multiplication, so it's a rank-1 term.

        # M1 = (A11 + A22)(B11 + B22)
        u1 = [0] * nm
        for pos in A11_pos:
            u1[pos] = 1
        for pos in A22_pos:
            u1[pos] = 1
        v1 = [0] * mp
        for pos in B11_pos:
            v1[pos] = 1
        for pos in B22_pos:
            v1[pos] = 1
        w1 = [0] * pn
        for pos in C11_pos:
            w1[pos] = 1
        for pos in C22_pos:
            w1[pos] = 1
        terms.append((u1, v1, w1))

        # M2 = (A21 + A22)B11
        u2 = [0] * nm
        for pos in A21_pos:
            u2[pos] = 1
        for pos in A22_pos:
            u2[pos] = 1
        v2 = [0] * mp
        for pos in B11_pos:
            v2[pos] = 1
        w2 = [0] * pn
        for pos in C21_pos:
            w2[pos] = 1
        for pos in C22_pos:
            w2[pos] = -1
        terms.append((u2, v2, w2))

        # M3 = A11(B21 - B22)
        u3 = [0] * nm
        for pos in A11_pos:
            u3[pos] = 1
        v3 = [0] * mp
        for pos in B21_pos:
            v3[pos] = 1
        for pos in B22_pos:
            v3[pos] = -1
        w3 = [0] * pn
        for pos in C12_pos:
            w3[pos] = 1
        for pos in C22_pos:
            w3[pos] = 1
        terms.append((u3, v3, w3))

        # M4 = A22(B12 - B11)
        u4 = [0] * nm
        for pos in A22_pos:
            u4[pos] = 1
        v4 = [0] * mp
        for pos in B12_pos:
            v4[pos] = 1
        for pos in B11_pos:
            v4[pos] = -1
        w4 = [0] * pn
        for pos in C11_pos:
            w4[pos] = 1
        for pos in C21_pos:
            w4[pos] = 1
        terms.append((u4, v4, w4))

        # M5 = (A11 + A12)B22
        u5 = [0] * nm
        for pos in A11_pos:
            u5[pos] = 1
        for pos in A12_pos:
            u5[pos] = 1
        v5 = [0] * mp
        for pos in B22_pos:
            v5[pos] = 1
        w5 = [0] * pn
        for pos in C11_pos:
            w5[pos] = -1
        for pos in C12_pos:
            w5[pos] = 1
        terms.append((u5, v5, w5))

        # M6 = (A21 - A11)(B11 + B12)
        u6 = [0] * nm
        for pos in A21_pos:
            u6[pos] = 1
        for pos in A11_pos:
            u6[pos] = -1
        v6 = [0] * mp
        for pos in B11_pos:
            v6[pos] = 1
        for pos in B12_pos:
            v6[pos] = 1
        w6 = [0] * pn
        for pos in C22_pos:
            w6[pos] = 1
        terms.append((u6, v6, w6))

        # M7 = (A12 - A22)(B21 + B22)
        u7 = [0] * nm
        for pos in A12_pos:
            u7[pos] = 1
        for pos in A22_pos:
            u7[pos] = -1
        v7 = [0] * mp
        for pos in B21_pos:
            v7[pos] = 1
        for pos in B22_pos:
            v7[pos] = 1
        w7 = [0] * pn
        for pos in C11_pos:
            w7[pos] = 1
        for pos in C12_pos:
            w7[pos] = -1
        terms.append((u7, v7, w7))

    # Now, we need to apply this to the 4x4x7 multiplication.
    # The 4x4 matrix A is divided into 2x2 blocks:
    # A = [[A11, A12], [A21, A22]]
    # The 4x7 matrix B is divided into 2x2 blocks (with the last block being 2x3):
    # B = [[B11, B12, B13], [B21, B22, B23]]
    # where B13 and B23 are 2x3 blocks.
    #
    # The product C = A * B is a 4x7 matrix:
    # C = [[C11, C12, C13], [C21, C22, C23]]
    # where C11 = A11*B11 + A12*B21, etc.
    #
    # We can use Strassen's algorithm to compute the 2x2 block products:
    # C11 = A11*B11 + A12*B21
    # C12 = A11*B12 + A12*B22
    # C13 = A11*B13 + A12*B23
    # C21 = A21*B11 + A22*B21
    # C22 = A21*B12 + A22*B22
    # C23 = A21*B13 + A22*B23
    #
    # Each of these is a sum of two 2x2x2 or 2x2x3 products.
    # We can use Strassen's algorithm for the 2x2x2 products, and schoolbook for the 2x2x3 products.
    #
    # But wait, the 2x2x3 product can be decomposed into 3 2x2x2 products (one for each column of the 2x3 block).
    # So we can use Strassen's algorithm for each of these 2x2x2 products.
    #
    # So for C13 = A11*B13 + A12*B23, we have:
    # A11*B13 is a 2x2x3 product, which we decompose into 3 2x2x2 products (one for each column of B13).
    # A12*B23 is a 2x2x3 product, which we decompose into 3 2x2x2 products (one for each column of B23).
    #
    # So for each 2x2x3 product, we need 3 * 7 = 21 rank-1 terms.
    #
    # Let's count the total number of terms:
    # C11 = A11*B11 + A12*B21: 2 * 7 = 14 terms
    # C12 = A11*B12 + A12*B22: 2 * 7 = 14 terms
    # C13 = A11*B13 + A12*B23: 2 * 21 = 42 terms
    # C21 = A21*B11 + A22*B21: 2 * 7 = 14 terms
    # C22 = A21*B12 + A22*B22: 2 * 7 = 14 terms
    # C23 = A21*B13 + A22*B23: 2 * 21 = 42 terms
    # Total: 14 + 14 + 42 + 14 + 14 + 42 = 140 terms.
    #
    # This is worse than schoolbook (112 terms). So this approach is not good.
    #
    # Let's try a different approach: use Strassen's algorithm for the 4x4x4 part (columns 0-3 of B),
    # and schoolbook for the 4x4x3 part (columns 4-6 of B).
    #
    # For the 4x4x4 part, we can use Strassen's algorithm for 4x4 matrix multiplication,
    # which has rank 49 (7^2).
    #
    # For the 4x4x3 part, we use schoolbook, which has rank 4*4*3 = 48.
    #
    # Total: 49 + 48 = 97 terms.
    #
    # Let's implement this.

    # First, clear the terms list
    terms = []

    # 4x4x4 part: columns 0-3 of B
    # We use Strassen's algorithm for 4x4 matrix multiplication.
    # The 4x4 matrix A is divided into 2x2 blocks:
    # A = [[A11, A12], [A21, A22]]
    # The 4x4 matrix B4 (columns 0-3 of B) is divided into 2x2 blocks:
    # B4 = [[B11, B12], [B21, B22]]
    # where B11 = B[:, 0:2], B12 = B[:, 2:4], etc.
    #
    # The product C4 = A * B4 is a 4x4 matrix:
    # C4 = [[C11, C12], [C21, C22]]
    #
    # We use Strassen's algorithm to compute C4.
    # The 7 Strassen products are:
    # M1 = (A11 + A22)(B11 + B22)
    # M2 = (A21 + A22)B11
    # M3 = A11(B21 - B22)
    # M4 = A22(B12 - B11)
    # M5 = (A11 + A12)B22
    # M6 = (A21 - A11)(B11 + B12)
    # M7 = (A12 - A22)(B21 + B22)
    #
    # Each Mi is a 2x2x2 product, which we decompose using 7 rank-1 terms.
    # So each Mi contributes 7 rank-1 terms, for a total of 7 * 7 = 49 terms.

    # Let's define the A_block entry positions for the 4x4x4 part:
    A11_pos = [0, 1, 4, 5]
    A12_pos = [2, 3, 6, 7]
    A21_pos = [8, 9, 12, 13]
    A22_pos = [10, 11, 14, 15]

    # B_block entry positions for the 4x4x4 part (columns 0-3 of B):
    B11_pos = [0, 1, 4, 5]      # B[:, 0:2]
    B12_pos = [2, 3, 6, 7]      # B[:, 2:4]
    B21_pos = [14, 15, 18, 19]  # B[2:4, 0:2]
    B22_pos = [16, 17, 20, 21]  # B[2:4, 2:4]

    # C_block entry positions for the 4x4x4 part:
    C11_pos = [0, 1, 4, 5]      # C[0:2, 0:2]
    C12_pos = [2, 3, 6, 7]      # C[0:2, 2:4]
    C21_pos = [14, 15, 18, 19]  # C[2:4, 0:2]
    C22_pos = [16, 17, 20, 21]  # C[2:4, 2:4]

    # Now, for each of the 7 Strassen products, we add 7 rank-1 terms.
    # Each rank-1 term corresponds to one of the 7 Strassen products of a 2x2x2 multiplication.

    def add_strassen_2x2_for_4x4(A_pos, B_pos, C_pos, w_offset):
        # A_pos: list of 4 lists, each list has 4 positions (for A11, A12, A21, A22)
        # B_pos: list of 4 lists, each list has 4 positions (for B11, B12, B21, B22)
        # C_pos: list of 4 lists, each list has 4 positions (for C11, C12, C21, C22)
        # w_offset: not used, since we're directly using the positions

        # M1 = (A11 + A22)(B11 + B22)
        u1 = [0] * nm
        for pos in A_pos[0]:  # A11
            u1[pos] = 1
        for pos in A_pos[3]:  # A22
            u1[pos] = 1
        v1 = [0] * mp
        for pos in B_pos[0]:  # B11
            v1[pos] = 1
        for pos in B_pos[3]:  # B22
            v1[pos] = 1
        w1 = [0] * pn
        for pos in C_pos[0]:  # C11
            w1[pos] = 1
        for pos in C_pos[3]:  # C22
            w1[pos] = 1
        terms.append((u1, v1, w1))

        # M2 = (A21 + A22)B11
        u2 = [0] * nm
        for pos in A_pos[2]:  # A21
            u2[pos] = 1
        for pos in A_pos[3]:  # A22
            u2[pos] = 1
        v2 = [0] * mp
        for pos in B_pos[0]:  # B11
            v2[pos] = 1
        w2 = [0] * pn
        for pos in C_pos[2]:  # C21
            w2[pos] = 1
        for pos in C_pos[3]:  # C22
            w2[pos] = -1
        terms.append((u2, v2, w2))

        # M3 = A11(B21 - B22)
        u3 = [0] * nm
        for pos in A_pos[0]:  # A11
            u3[pos] = 1
        v3 = [0] * mp
        for pos in B_pos[2]:  # B21
            v3[pos] = 1
        for pos in B_pos[3]:  # B22
            v3[pos] = -1
        w3 = [0] * pn
        for pos in C_pos[1]:  # C12
            w3[pos] = 1
        for pos in C_pos[3]:  # C22
            w3[pos] = 1
        terms.append((u3, v3, w3))

        # M4 = A22(B12 - B11)
        u4 = [0] * nm
        for pos in A_pos[3]:  # A22
            u4[pos] = 1
        v4 = [0] * mp
        for pos in B_pos[1]:  # B12
            v4[pos] = 1
        for pos in B_pos[0]:  # B11
            v4[pos] = -1
        w4 = [0] * pn
        for pos in C_pos[0]:  # C11
            w4[pos] = 1
        for pos in C_pos[2]:  # C21
            w4[pos] = 1
        terms.append((u4, v4, w4))

        # M5 = (A11 + A12)B22
        u5 = [0] * nm
        for pos in A_pos[0]:  # A11
            u5[pos] = 1
        for pos in A_pos[1]:  # A12
            u5[pos] = 1
        v5 = [0] * mp
        for pos in B_pos[3]:  # B22
            v5[pos] = 1
        w5 = [0] * pn
        for pos in C_pos[0]:  # C11
            w5[pos] = -1
        for pos in C_pos[1]:  # C12
            w5[pos] = 1
        terms.append((u5, v5, w5))

        # M6 = (A21 - A11)(B11 + B12)
        u6 = [0] * nm
        for pos in A_pos[2]:  # A21
            u6[pos] = 1
        for pos in A_pos[0]:  # A11
            u6[pos] = -1
        v6 = [0] * mp
        for pos in B_pos[0]:  # B11
            v6[pos] = 1
        for pos in B_pos[1]:  # B12
            v6[pos] = 1
        w6 = [0] * pn
        for pos in C_pos[3]:  # C22
            w6[pos] = 1
        terms.append((u6, v6, w6))

        # M7 = (A12 - A22)(B21 + B22)
        u7 = [0] * nm
        for pos in A_pos[1]:  # A12
            u7[pos] = 1
        for pos in A_pos[3]:  # A22
            u7[pos] = -1
        v7 = [0] * mp
        for pos in B_pos[2]:  # B21
            v7[pos] = 1
        for pos in B_pos[3]:  # B22
            v7[pos] = 1
        w7 = [0] * pn
        for pos in C_pos[0]:  # C11
            w7[pos] = 1
        for pos in C_pos[1]:  # C12
            w7[pos] = -1
        terms.append((u7, v7, w7))

    # Call add_strassen_2x2_for_4x4 for the 4x4x4 part
    A_pos = [A11_pos, A12_pos, A21_pos, A22_pos]
    B_pos = [B11_pos, B12_pos, B21_pos, B22_pos]
    C_pos = [C11_pos, C12_pos, C21_pos, C22_pos]
    add_strassen_2x2_for_4x4(A_pos, B_pos, C_pos, 0)

    # Now, for the 4x4x3 part: columns 4-6 of B
    # We use schoolbook for this part.
    # The 4x4 matrix A is the same as before.
    # The 4x3 matrix B3 (columns 4-6 of B) is:
    # B3 = [[B13], [B23]]
    # where B13 = B[:, 4:7] and B23 = B[2:4, 4:7].
    #
    # The product C3 = A * B3 is a 4x3 matrix:
    # C3 = [[C13], [C23]]
    # where C13 = A11*B13 + A12*B23 and C23 = A21*B13 + A22*B23.
    #
    # We use schoolbook for this part:
    for i in range(n):
        for j in range(m):
            for k in range(3):  # columns 4, 5, 6 of B
                u = make_u(i, j, 1)
                v = make_v(j, 4 + k, 1)  # columns 4, 5, 6 of B
                w = make_w(4 + k, i, 1)  # columns 4, 5, 6 of C
                terms.append((u, v, w))

    rank = len(terms)
    U = [[0] * rank for _ in range(nm)]
    V = [[0] * rank for _ in range(mp)]
    W = [[0] * rank for _ in range(pn)]

    for c, (u, v, w) in enumerate(terms):
        for r in range(nm):
            U[r][c] = u[r]
        for r in range(mp):
            V[r][c] = v[r]
        for r in range(pn):
            W[r][c] = w[r]

    return {"U": U, "V": V, "W": W}
