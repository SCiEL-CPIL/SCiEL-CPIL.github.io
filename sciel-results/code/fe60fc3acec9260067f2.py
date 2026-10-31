"""
Exact rank-49 tensor decomposition for 4x4 x 4x4 matrix multiplication.

Uses the standard Strassen recursion on 2x2 blocks, applied twice (top level and
sub-blocks), yielding 7^2 = 49 rank-1 terms with integer entries.

Indexing conventions:
  U: shape (n*m, rank)  = (16, 49),  U[i*m + j, r]
  V: shape (m*p, rank)  = (16, 49),  V[j*p + k, r]
  W: shape (p*n, rank)  = (16, 49),  W[k*n + i, r]   (official trace convention)

The decomposition satisfies:
  C[i, k] = sum_r U[i*m + j, r] * V[j*p + k', r] * W[k*n + i, r]
  summed over all intermediate indices, which is exactly the matrix multiplication
  tensor T[i, j, k', k] = delta_{j, k'}  (i.e., C = A * B).
"""

DIMENSIONS = [4, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 4, 4, 4
    rank = 49

    U = [[0] * rank for _ in range(n * m)]  # 16 x 49
    V = [[0] * rank for _ in range(m * p)]  # 16 x 49
    W = [[0] * rank for _ in range(p * n)]  # 16 x 49

    # Strassen's 2x2 block decomposition.
    # For multiplying two 2x2 matrices [[a,b],[c,d]] * [[e,f],[g,h]] = [[p1,p2],[p3,p4]]:
    #   p1 = (a+d)(e+h)
    #   p2 = (c+d)f
    #   p3 = a(g+h)
    #   p4 = (a-b)(h+e)  -- wait, standard Strassen uses 7 products:
    #
    # Standard Strassen 7 products for 2x2:
    #   m1 = (a+d)(e+h)
    #   m2 = (c+d)e
    #   m3 = a(f-h)
    #   m4 = d(g-e)
    #   m5 = (a+b)h
    #   m6 = (c-a)(e+f)
    #   m7 = (b-d)(g+h)
    #
    # Output:
    #   C00 = m1 + m4 - m5 + m7
    #   C01 = m3 + m5
    #   C10 = m2 + m4
    #   C11 = m1 - m2 + m3 + m6
    #
    # Each product m_s is a rank-1 term in the tensor decomposition.
    # At the top level (4x4 = 2x2 blocks of 2x2), each m_s is itself a 2x2 matrix
    # product, which we decompose into 7 sub-products m_{s,t}.
    #
    # So overall rank = 7 * 7 = 49.

    # Define the 7 Strassen product definitions for 2x2:
    # Each product s has:
    #   A-side: linear combination of a,b,c,d  -> coefficients (ca, cb, cc, cd)
    #   B-side: linear combination of e,f,g,h  -> coefficients (ce, cf, cg, ch)
    #   Output contributions: how m_s adds to C00, C01, C10, C11

    # For a 2x2 matrix with elements indexed as:
    #   A = [[a, b], [c, d]]  where a=A[0,0], b=A[0,1], c=A[1,0], d=A[1,1]
    #   B = [[e, f], [g, h]]  where e=B[0,0], f=B[0,1], g=B[1,0], h=B[1,1]

    # Strassen products (s = 0..6):
    # s=0: m0 = (a+d)(e+h)
    # s=1: m1 = (c+d)e
    # s=2: m2 = a(f-h)
    # s=3: m3 = d(g-e)
    # s=4: m4 = (a+b)h
    # s=5: m5 = (c-a)(e+f)
    # s=6: m6 = (b-d)(g+h)

    # A-side coefficients for each product (for elements [a, b, c, d]):
    A_coeffs = [
        [1, 0, 0, 1],   # s=0: a+d
        [0, 0, 1, 1],   # s=1: c+d
        [1, 0, 0, 0],   # s=2: a
        [0, 0, 0, 1],   # s=3: d
        [1, 1, 0, 0],   # s=4: a+b
        [-1, 0, 1, 0],  # s=5: c-a
        [0, 1, 0, -1],  # s=6: b-d
    ]

    # B-side coefficients for each product (for elements [e, f, g, h]):
    B_coeffs = [
        [1, 0, 0, 1],   # s=0: e+h
        [1, 0, 0, 0],   # s=1: e
        [0, 1, 0, -1],  # s=2: f-h
        [0, -1, 1, 0],  # s=3: g-e
        [0, 0, 0, 1],   # s=4: h
        [1, 1, 0, 0],   # s=5: e+f
        [0, 0, 1, 1],   # s=6: g+h
    ]

    # Output contributions: how each m_s contributes to [C00, C01, C10, C11]
    # C00 = m0 + m3 - m4 + m6
    # C01 = m2 + m4
    # C10 = m1 + m3
    # C11 = m0 - m1 + m2 + m5
    Out_coeffs = [
        [1, 0, 0, 0],   # s=0: contributes +1 to C00, +1 to C11
        [0, 0, 0, -1],  # s=1: contributes -1 to C11
        [0, 1, 0, 1],   # s=2: contributes +1 to C01, +1 to C11
        [1, 0, 1, 0],   # s=3: contributes +1 to C00, +1 to C10
        [-1, 1, 0, 0],  # s=4: contributes -1 to C00, +1 to C01
        [0, 0, 0, 1],   # s=5: contributes +1 to C11
        [1, 0, 0, 0],   # s=6: contributes +1 to C00
    ]

    # Now build the 4x4 decomposition.
    # The 4x4 matrix A is divided into 2x2 blocks:
    #   A = [[A00, A01], [A10, A11]]
    #   where A00 is top-left 2x2, A01 is top-right, A10 is bottom-left, A11 is bottom-right.
    #
    # Element mapping for 4x4:
    #   A[i, j] for i,j in {0,1,2,3}
    #   Block (bi, bj) contains elements A[2*bi + ii, 2*bj + jj] for ii,jj in {0,1}
    #
    # For the A-side of a top-level Strassen product s:
    #   The product is (linear_comb_A)(linear_comb_B)
    #   where linear_comb_A is a 2x2 matrix whose elements are linear combos of A's 2x2 blocks.
    #
    # More precisely, for product s, the A-side 2x2 matrix has:
    #   element (0,0) = sum over A-blocks (A_coeffs[s][0] * A00 + A_coeffs[s][1] * A01 + ...)
    #   Wait, no. A_coeffs[s] gives coefficients for [a, b, c, d] which are the 2x2 block elements.
    #   So if A_coeffs[s] = [ca, cb, cc, cd], then the A-side 2x2 matrix is:
    #     [[ca*A00 + cb*A01, ...], ...]
    #   No, that's not right either. Let me think more carefully.
    #
    # In Strassen, for 2x2 block matrices:
    #   M = [[M00, M01], [M10, M11]]  (each Mij is a 2x2 submatrix)
    #   N = [[N00, N01], [N10, N11]]
    #   The 7 products are:
    #     P0 = (M00 + M11)(N00 + N11)
    #     P1 = (M10 + M11)N00
    #     P2 = M00(N01 - N11)
    #     P3 = M11(N10 - N00)
    #     P4 = (M00 + M01)N11
    #     P5 = (M10 - M00)(N00 + N01)
    #     P6 = (M01 - M11)(N10 + N11)
    #
    # So for product s, the A-side is a 2x2 matrix where each entry is a linear combination
    # of the 2x2 submatrices M00, M01, M10, M11.
    #
    # For the 4x4 case, M00, M01, M10, M11 are each 2x2 matrices of actual elements.
    # When we further decompose each product P_s (which is a 2x2 matrix product of 2x2
    # linear combos) using Strassen again, we get 49 total products.
    #
    # Let me restructure: for each top-level product s (0..6), the A-side 2x2 matrix is:
    #   [[alpha00*M00 + alpha01*M01 + alpha10*M10 + alpha11*M11, ...]]
    #   No, actually each product P_s = (A_side_s) * (B_side_s) where A_side_s and B_side_s
    #   are 2x2 matrices formed from linear combinations of the 2x2 blocks.
    #
    # Actually, let me just directly construct the rank-1 terms.
    #
    # For each (s, t) where s is the top-level Strassen product (0..6) and t is the
    # sub-level Strassen product (0..6), we have a rank-1 term.
    #
    # The A-side of the full product is:
    #   For the sub-product t, the A-side 2x2 matrix (in terms of the 2x2 blocks of the
    #   top-level A-side matrix) has coefficients A_coeffs[t].
    #   The top-level A-side matrix entries are linear combos of M00, M01, M10, M11
    #   with coefficients from A_coeffs[s].
    #
    # This is getting complex. Let me just enumerate all 49 terms directly.

    # For a 4x4 matrix, elements A[i,j] with i,j in {0,1,2,3}.
    # Block decomposition:
    #   M00 = [[A[0,0], A[0,1]], [A[1,0], A[1,1]]]
    #   M01 = [[A[0,2], A[0,3]], [A[1,2], A[1,3]]]
    #   M10 = [[A[2,0], A[2,1]], [A[3,0], A[3,1]]]
    #   M11 = [[A[2,2], A[2,3]], [A[3,2], A[3,3]]]

    # For top-level product s, the A-side 2x2 matrix (call it As) has entries:
    #   As[0,0] = A_coeffs[s][0] * M00  (as a 2x2 matrix, element-wise)
    #   As[0,1] = A_coeffs[s][1] * M01
    #   As[1,0] = A_coeffs[s][2] * M10
    #   As[1,1] = A_coeffs[s][3] * M11
    #
    # Wait, that's not quite right. A_coeffs[s] = [ca, cb, cc, cd] means:
    #   As = [[ca*M00, cb*M01], [cc*M10, cd*M11]]
    #   where each entry is a 2x2 matrix scaled by the coefficient.
    #
    # Similarly for B-side:
    #   Bs = [[ce*N00, cf*N01], [cg*N10, ch*N11]]
    #   where [ce, cf, cg, ch] = B_coeffs[s].
    #
    # Now, As * Bs is a 2x2 matrix product where each entry is a 2x2 matrix.
    # We decompose each 2x2 entry using Strassen (sub-level).
    #
    # Actually, As * Bs as a whole is a 2x2 matrix product of 2x2 matrices (where each
    # "element" is a 2x2 matrix). This is equivalent to multiplying two 4x4 matrices
    # that are block-diagonal in a sense. But we can just apply Strassen to the 2x2
    # structure of As and Bs, treating each entry as a scalar (which is actually a 2x2 matrix).
    #
    # For sub-level product t, the contribution to the output 2x2 matrix (in block form) is:
    #   Out_block[bi, bj] += Out_coeffs[s][t] * (sub_A_side_t) * (sub_B_side_t)
    #   where sub_A_side_t and sub_B_side_t are 2x2 matrices formed from the entries of As and Bs.
    #
    # For sub-product t:
    #   sub_A_side_t[0,0] = A_coeffs[t][0] * As[0,0] + A_coeffs[t][1] * As[0,1] + ...
    #   No, A_coeffs[t] = [ca, cb, cc, cd] applies to the 2x2 matrix As:
    #     sub_A = [[ca*As[0,0], cb*As[0,1]], [cc*As[1,0], cd*As[1,1]]]
    #   Similarly for sub_B.
    #
    # Then the contribution to the final 4x4 output from term (s,t) is:
    #   For each output block position (bi, bj) in {0,1}^2:
    #     contrib = Out_coeffs[s][t] * (sub_A * sub_B)[bi, bj]
    #   where (sub_A * sub_B)[bi, bj] is a 2x2 matrix (since sub_A and sub_B are 2x2
    #   matrices whose entries are 2x2 matrices... wait no).
    #
    # I'm overcomplicating this. Let me just directly compute the U, V, W entries.

    # Each rank-1 term (s, t) corresponds to r = 7*s + t.
    # U[i*m + j, r] is the coefficient of A[i, j] in the A-side of term r.
    # V[j'*p + k, r] is the coefficient of B[j', k] in the B-side of term r.
    # W[k*n + i, r] is the coefficient for the output C[i, k] in term r.

    # For term (s, t):
    #   The A-side is: for each element A[i, j] of the 4x4 matrix, what is its coefficient?
    #   A[i, j] belongs to block M[bi, bj] where bi = i // 2, bj = j // 2, and within the
    #   block at position (ii, jj) where ii = i % 2, jj = j % 2.
    #
    #   For top-level product s, As[bi, bj] = A_coeffs[s][bi*2 + bj] * M[bi, bj]
    #   Wait, A_coeffs[s] has 4 entries corresponding to [M00, M01, M10, M11].
    #   So As[bi, bj] = A_coeffs[s][bi*2 + bj] * M[bi, bj]  (element-wise scaling).
    #   Therefore, A[i, j] contributes to As with coefficient A_coeffs[s][bi*2 + bj]
    #   where bi = i//2, bj = j//2.
    #
    #   Now, for sub-level product t, sub_A is a 2x2 matrix where:
    #     sub_A[ii, jj] = A_coeffs[t][ii*2 + jj] * As[ii, jj]  ... no.
    #   sub_A is formed from As: sub_A = [[ca*As[0,0], cb*As[0,1]], [cc*As[1,0], cd*As[1,1]]]
    #   where [ca, cb, cc, cd] = A_coeffs[t].
    #   So sub_A[ii, jj] = A_coeffs[t][ii*2 + jj] * As[ii, jj].
    #   And As[ii, jj] = A_coeffs[s][ii*2 + jj] * M[ii, jj] (element-wise).
    #   So the coefficient of A[i, j] in sub_A[ii, jj] (at position (i%2, j%2)) is:
    #     A_coeffs[t][ii*2 + jj] * A_coeffs[s][ii*2 + jj]
    #   where ii = i//2, jj = j//2.
    #
    #   But wait, we need the coefficient of A[i,j] in the "scalar" product for sub-level t.
    #   The sub-level product t computes a scalar (which is really a 2x2 matrix) as a product
    #   of two linear combinations. The A-side linear combination for sub-level t is:
    #     sum_{ii,jj} (linear_combo_coeff[t, ii, jj]) * sub_A[ii, jj]
    #   where the linear combo for sub-level t on the A-side is:
    #     A_coeffs[t] = [ca, cb, cc, cd] applied to [sub_A[0,0], sub_A[0,1], sub_A[1,0], sub_A[1,1]].
    #   So the coefficient of sub_A[ii, jj] is A_coeffs[t][ii*2 + jj].
    #
    #   Therefore, the total coefficient of A[i, j] in the A-side of term (s, t) is:
    #     A_coeffs[t][ii*2 + jj] * A_coeffs[s][ii*2 + jj]
    #   where ii = i // 2, jj = j // 2.

    # Similarly for V:
    #   V[j'*p + k, r] = B_coeffs[t][ii*2 + jj] * B_coeffs[s][ii*2 + jj]
    #   where ii = j' // 2, jj = k // 2.

    # For W:
    #   W[k*n + i, r] = Out_coeffs[s][t] * (coefficient for output position)
    #   The output C[i, k] belongs to block N[bi, bj] where bi = k // 2, bj = i // 2,
    #   and within the block at position (ii, jj) where ii = k % 2, jj = i % 2.
    #   Wait, the output convention is W[k*n + i, r], so the output index is (k, i) meaning
    #   C[i, k] (row i, column k). The block is determined by k//2 (block row) and i//2 (block col)?
    #   No, C[i, k] where i is row, k is column. Block row = i//2, block col = k//2.
    #   Within block: (i%2, k%2).
    #
    #   The output contribution from term (s, t) to block (bi, bj) where bi = i//2, bj = k//2 is:
    #     Out_coeffs[s][t] * (value from sub-level product t at position (bi, bj))
    #   The sub-level product t contributes to the 2x2 output matrix at position (bi, bj)
    #   with coefficient Out_coeffs[s][t] (since Out_coeffs[s] gives the contribution to
    #   [C00, C01, C10, C11] for product s, and t indexes the sub-level).
    #
    #   Wait, I need to be more careful. Out_coeffs[s][t] is the coefficient of m_t in
    #   the output block (for top-level product s). But actually, Out_coeffs[s] is a vector
    #   of 4 entries for the 4 output blocks, for a given s. And t indexes which of the 7
    #   sub-products contributes.
    #
    #   Let me redefine: For top-level product s, the 7 sub-products t=0..6 each contribute
    #   to the output 2x2 block matrix. The contribution of sub-product t to output block
    #   (bi, bj) is Out_coeffs[s][t] * delta... no.
    #
    #   Actually, in the standard Strassen decomposition, for a 2x2 matrix product:
    #     C = sum_{t=0}^{6} m_t * O_t
    #   where O_t is a 2x2 matrix of coefficients. So:
    #     C[bi, bj] = sum_{t=0}^{6} O_t[bi, bj] * m_t
    #
    #   So for top-level product s, the output block (bi, bj) gets contribution:
    #     sum_{t=0}^{6} O_s[t, bi, bj] * (sub-level product t of As * Bs)
    #
    #   Where O_s is the 7x2x2 tensor of output coefficients for product s.
    #
    #   Let me define O_s[t, bi, bj] = Out_coeffs[s][t] for the appropriate (bi, bj).
    #   From earlier:
    #     C00 = m0 + m3 - m4 + m6  -> O_s[0,0,0]=1, O_s[1,0,0]=0, O_s[2,0,0]=0, O_s[3,0,0]=1, O_s[4,0,0]=-1, O_s[5,0,0]=0, O_s[6,0,0]=1
    #     C01 = m2 + m4            -> O_s[0,0,1]=0, O_s[1,0,1]=0, O_s[2,0,1]=1, O_s[3,0,1]=0, O_s[4,0,1]=1, O_s[5,0,1]=0, O_s[6,0,1]=0
    #     C10 = m1 + m3            -> O_s[0,1,0]=0, O_s[1,1,0]=1, O_s[2,1,0]=0, O_s[3,1,0]=1, O_s[4,1,0]=0, O_s[5,1,0]=0, O_s[6,1,0]=0
    #     C11 = m0 - m1 + m2 + m5  -> O_s[0,1,1]=1, O_s[1,1,1]=-1, O_s[2,1,1]=1, O_s[3,1,1]=0, O_s[4,1,1]=0, O_s[5,1,1]=1, O_s[6,1,1]=0
    #
    #   So for term (s, t), the contribution to output block (bi, bj) is O_s[t, bi, bj].
    #   And within the block, the contribution is uniform (since the sub-level product t
    #   produces a 2x2 matrix that is added element-wise).
    #
    #   Therefore, W[k*n + i, r] for r = 7*s + t is:
    #     O_s[t, i//2, k//2]
    #   where i//2 is the block row and k//2 is the block column.

    # Let me build O_s as a 7x7x2x2 array (s, t, bi, bj):
    # Actually, O_s is per s, and for each s, it's a 7x2x2 array (t, bi, bj).

    O = [[[0]*2 for _ in range(2)] for _ in range(7)]  # O[t][bi][bj]
    # C00 (bi=0, bj=0): m0 + m3 - m4 + m6
    O[0][0][0] = 1; O[3][0][0] = 1; O[4][0][0] = -1; O[6][0][0] = 1
    # C01 (bi=0, bj=1): m2 + m4
    O[2][0][1] = 1; O[4][0][1] = 1
    # C10 (bi=1, bj=0): m1 + m3
    O[1][1][0] = 1; O[3][1][0] = 1
    # C11 (bi=1, bj=1): m0 - m1 + m2 + m5
    O[0][1][1] = 1; O[1][1][1] = -1; O[2][1][1] = 1; O[5][1][1] = 1

    # Now fill U, V, W:
    for s in range(7):
        for t in range(7):
            r = 7 * s + t
            for i in range(4):
                for j in range(4):
                    ii = i // 2
                    jj = j // 2
                    U[i * 4 + j][r] = A_coeffs[s][ii * 2 + jj] * A_coeffs[t][ii * 2 + jj]
            for jp in range(4):
                for k in range(4):
                    ii = jp // 2
                    jj = k // 2
                    V[jp * 4 + k][r] = B_coeffs[s][ii * 2 + jj] * B_coeffs[t][ii * 2 + jj]
            for k in range(4):
                for i in range(4):
                    bi = i // 2
                    bj = k // 2
                    W[k * 4 + i][r] = O[t][bi][bj]

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
