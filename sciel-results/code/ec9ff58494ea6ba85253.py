"""4x4x4 matrix multiplication tensor decomposition, rank 49 (Bini et al.)."""
DIMENSIONS = [4, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank = 49
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    # Strassen's 7 products for 2x2 matrices.
    # Given A = [[a,b],[c,d]], B = [[e,f],[g,h]]:
    #   P1 = (a+d)(e+h)
    #   P2 = (c+d)e
    #   P3 = a(f-h)
    #   P4 = d(g-e)
    #   P5 = (a+b)h
    #   P6 = (c-a)(e+f)
    #   P7 = (d-b)(g+h)
    # Result: [[P1+P4-P5+P7, P3+P5], [P2+P4, P1-P2+P3+P6]]

    # Strassen's 7 products for scalar (1x1) "matrices" -- just the 7 linear forms:
    # Each scalar product r is a product of two linear combinations.
    # For a single scalar product, we define the "scalar Strassen" as:
    #   s1 = (a+d)(e+h)  -> but for scalars a=d, b=c, e=h, f=g... 
    # Actually, for 1x1 there's no Strassen reduction. Instead, we use
    # the 7-term structure differently.
    
    # The 4x4 decomposition: split into 2x2 blocks.
    # A = [[A00, A01], [A10, A11]] where each Aij is 2x2.
    # B = [[B00, B01], [B10, B11]] where each Bij is 2x2.
    # C = [[C00, C01], [C10, C11]] where each Cij is 2x2.
    #
    # Strassen at the block level (7 block products):
    #   S1 = (A00+A11)(B00+B11)
    #   S2 = (A10+A11)B00
    #   S3 = A00(B01-B11)
    #   S4 = A11(B10-B00)
    #   S5 = (A00+A01)B11
    #   S6 = (A10-A00)(B00+B01)
    #   S7 = (A11-A01)(B10+B11)
    #
    # C00 = S1 + S4 - S5 + S7
    # C01 = S3 + S5
    # C10 = S2 + S4
    # C11 = S1 - S2 + S3 + S6
    #
    # Each Sx is a 2x2 matrix product, computed via Strassen (7 scalar products).
    # Total: 7 * 7 = 49 scalar products.

    # For each of the 7 block-level Strassen products, we need to express
    # the two 2x2 matrices being multiplied as linear combinations of the
    # entries of A and B, then apply Strassen to get 7 scalar products.

    # Strassen for 2x2: given matrices X = [[x0,x1],[x2,x3]], Y = [[y0,y1],[y2,y3]]
    # 7 products:
    #   r1 = (x0+x3)(y0+y3)
    #   r2 = (x2+x3)y0
    #   r3 = x0(y1-y3)
    #   r4 = x3(y2-y0)
    #   r5 = (x0+x1)y3
    #   r6 = (x2-x0)(y0+y1)
    #   r7 = (x3-x1)(y2+y3)
    # Result: [[r1+r4-r5+r7, r3+r5], [r2+r4, r1-r2+r3+r6]]

    # We'll build the 49-term decomposition by iterating over the 7 block
    # products and, for each, the 7 scalar Strassen products.

    # Define the 7 block-level linear forms for A and B.
    # For block product t (0..6), the A-side is a 2x2 matrix whose entries
    # are linear combinations of A's 16 entries, and similarly for B.

    # A block indices: A[bi][bj] where bi in {0,1}, bj in {0,1}, each 2x2.
    # A entry (i,j) belongs to block (i//2, j//2) at position (i%2, j%2).

    # For each block product t, define:
    #   A_t = [[A_t00, A_t01], [A_t10, A_t11]] (2x2 matrix of linear forms)
    #   B_t = [[B_t00, B_t01], [B_t10, B_t11]] (2x2 matrix of linear forms)

    # The 7 block products:
    # t=0: A_t = A00+A11, B_t = B00+B11
    # t=1: A_t = A10+A11, B_t = B00
    # t=2: A_t = A00, B_t = B01-B11
    # t=3: A_t = A11, B_t = B10-B00
    # t=4: A_t = A00+A01, B_t = B11
    # t=5: A_t = A10-A00, B_t = B00+B01
    # t=6: A_t = A11-A01, B_t = B10+B11

    # Each block (e.g., A00) is a 2x2 matrix: A00 = [[A[0,0], A[0,1]], [A[1,0], A[1,1]]]
    # So A_t is a 2x2 matrix where each entry is a sum/difference of specific A entries.

    # Let's define, for each t, the coefficients of A_t and B_t.
    # A_t[r][c] = sum of coeff * A[i][j] for the 4 entries in the relevant blocks.

    # Represent each 2x2 block's entries by their (i,j) positions in the 4x4 matrix.
    # Block (bi, bj) has entries at (2*bi, 2*bj), (2*bi, 2*bj+1), (2*bi+1, 2*bj), (2*bi+1, 2*bj+1)

    # For block product t, A_t is a 2x2 matrix. Its (r,c) entry is a linear form.
    # We represent A_t[r][c] as a dict: {(i,j): coeff}

    def make_block(bi, bj, sign=1):
        """Return a dict {(i,j): coeff} for the 2x2 block (bi,bj) with given sign."""
        result = {}
        for r in range(2):
            for c in range(2):
                i = 2 * bi + r
                j = 2 * bj + c
                result[(i, j)] = sign
        return result

    def add_blocks(d1, d2):
        """Add two block dicts."""
        result = dict(d1)
        for key, val in d2.items():
            result[key] = result.get(key, 0) + val
        return result

    # Define the 7 A-side and B-side 2x2 matrices for the block-level Strassen.
    # Each is a 2x2 matrix of block-dicts.

    # t=0: A_t = A00+A11, B_t = B00+B11
    # A_t[r][c] = A00[r][c] + A11[r][c]
    A_t0 = [[add_blocks(make_block(0,0), make_block(1,1)) for c in range(2)] for r in range(2)]
    B_t0 = [[add_blocks(make_block(0,0), make_block(1,1)) for c in range(2)] for r in range(2)]

    # t=1: A_t = A10+A11, B_t = B00
    A_t1 = [[add_blocks(make_block(1,0), make_block(1,1)) for c in range(2)] for r in range(2)]
    B_t1 = [[make_block(0,0) for c in range(2)] for r in range(2)]

    # t=2: A_t = A00, B_t = B01-B11
    A_t2 = [[make_block(0,0) for c in range(2)] for r in range(2)]
    B_t2 = [[add_blocks(make_block(0,1), make_block(1,1, -1)) for c in range(2)] for r in range(2)]

    # t=3: A_t = A11, B_t = B10-B00
    A_t3 = [[make_block(1,1) for c in range(2)] for r in range(2)]
    B_t3 = [[add_blocks(make_block(1,0), make_block(0,0, -1)) for c in range(2)] for r in range(2)]

    # t=4: A_t = A00+A01, B_t = B11
    A_t4 = [[add_blocks(make_block(0,0), make_block(0,1)) for c in range(2)] for r in range(2)]
    B_t4 = [[make_block(1,1) for c in range(2)] for r in range(2)]

    # t=5: A_t = A10-A00, B_t = B00+B01
    A_t5 = [[add_blocks(make_block(1,0), make_block(0,0, -1)) for c in range(2)] for r in range(2)]
    B_t5 = [[add_blocks(make_block(0,0), make_block(0,1)) for c in range(2)] for r in range(2)]

    # t=6: A_t = A11-A01, B_t = B10+B11
    A_t6 = [[add_blocks(make_block(1,1), make_block(0,1, -1)) for c in range(2)] for r in range(2)]
    B_t6 = [[add_blocks(make_block(1,0), make_block(1,1)) for c in range(2)] for r in range(2)]

    A_sides = [A_t0, A_t1, A_t2, A_t3, A_t4, A_t5, A_t6]
    B_sides = [B_t0, B_t1, B_t2, B_t3, B_t4, B_t5, B_t6]

    # Now, for each block product t, apply Strassen to the 2x2 matrices A_t and B_t.
    # Strassen's 7 products for 2x2:
    #   r1 = (x0+x3)(y0+y3)
    #   r2 = (x2+x3)y0
    #   r3 = x0(y1-y3)
    #   r4 = x3(y2-y0)
    #   r5 = (x0+x1)y3
    #   r6 = (x2-x0)(y0+y1)
    #   r7 = (x3-x1)(y2+y3)

    # For each of the 7 scalar products, we need:
    #   - The A-side linear form (for U)
    #   - The B-side linear form (for V)
    #   - The C-side contribution (for W)

    # The result of the block product t contributes to C blocks as:
    # t=0: C00 += S1, C11 += S1
    # t=1: C10 += S2, C11 -= S2
    # t=2: C01 += S3, C11 += S3
    # t=3: C00 += S4, C10 += S4
    # t=4: C00 -= S5, C01 += S5
    # t=5: C11 += S6
    # t=6: C00 += S7

    # More precisely:
    # C00 = S1 + S4 - S5 + S7
    # C01 = S3 + S5
    # C10 = S2 + S4
    # C11 = S1 - S2 + S3 + S6

    # So for block product t, the contribution to C is:
    # t=0 (S1): C00 += 1, C11 += 1
    # t=1 (S2): C10 += 1, C11 -= 1
    # t=2 (S3): C01 += 1, C11 += 1
    # t=3 (S4): C00 += 1, C10 += 1
    # t=4 (S5): C00 -= 1, C01 += 1
    # t=5 (S6): C11 += 1
    # t=6 (S7): C00 += 1

    # For each block product t and each Strassen scalar product s (0..6),
    # the C-side is a 2x2 matrix (the Strassen result for that scalar product)
    # multiplied by the block-level coefficient.

    # Strassen result for scalar product s:
    # s=0 (r1): [[1, 0], [0, 1]] (contributes to both diagonal)
    # Actually, let's be more careful. The Strassen formula gives:
    #   Result[0][0] = r1+r4-r5+r7
    #   Result[0][1] = r3+r5
    #   Result[1][0] = r2+r4
    #   Result[1][1] = r1-r2+r3+r6
    #
    # So each scalar product r_s contributes to specific C entries:
    # s=0 (r1): C[0][0] += 1, C[1][1] += 1
    # s=1 (r2): C[1][0] += 1, C[1][1] -= 1
    # s=2 (r3): C[0][1] += 1, C[1][1] += 1
    # s=3 (r4): C[0][0] += 1, C[1][0] += 1
    # s=4 (r5): C[0][0] -= 1, C[0][1] += 1
    # s=5 (r6): C[1][1] += 1
    # s=6 (r7): C[0][0] += 1

    # For the 2x2 Strassen within a block product, the "x" and "y" are the
    # 2x2 matrices A_t and B_t. The scalar products are:
    # s=0: x0+x3 times y0+y3
    # s=1: (x2+x3) times y0
    # s=2: x0 times (y1-y3)
    # s=3: x3 times (y2-y0)
    # s=4: (x0+x1) times y3
    # s=5: (x2-x0) times (y0+y1)
    # s=6: (x3-x1) times (y2+y3)

    # Where x0=A_t[0][0], x1=A_t[0][1], x2=A_t[1][0], x3=A_t[1][1], etc.

    # For each scalar product s, the A-side linear form is a sum of entries of A_t,
    # which is a sum of entries of the original A matrix.

    # Let's define the A-side and B-side linear forms for each (t, s).

    # For s=0: A-side = x0+x3 = A_t[0][0] + A_t[1][1]
    #          B-side = y0+y3 = B_t[0][0] + B_t[1][1]
    # For s=1: A-side = x2+x3 = A_t[1][0] + A_t[1][1]
    #          B-side = y0 = B_t[0][0]
    # For s=2: A-side = x0 = A_t[0][0]
    #          B-side = y1-y3 = B_t[0][1] - B_t[1][1]
    # For s=3: A-side = x3 = A_t[1][1]
    #          B-side = y2-y0 = B_t[1][0] - B_t[0][0]
    # For s=4: A-side = x0+x1 = A_t[0][0] + A_t[0][1]
    #          B-side = y3 = B_t[1][1]
    # For s=5: A-side = x2-x0 = A_t[1][0] - A_t[0][0]
    #          B-side = y0+y1 = B_t[0][0] + B_t[0][1]
    # For s=6: A-side = x3-x1 = A_t[1][1] - A_t[0][1]
    #          B-side = y2+y3 = B_t[1][0] + B_t[1][1]

    # The C-side for each (t, s) is a 2x2 matrix of coefficients, which when
    # combined with the block-level contribution gives the final C coefficients.

    # C-side for Strassen scalar product s (within a block product):
    # s=0: [[1, 0], [0, 1]]
    # s=1: [[0, 0], [1, -1]]
    # s=2: [[0, 1], [0, 1]]
    # s=3: [[1, 0], [1, 0]]
    # s=4: [[-1, 1], [0, 0]]
    # s=5: [[0, 0], [0, 1]]
    # s=6: [[1, 0], [0, 0]]

    C_strassen = [
        [[1, 0], [0, 1]],   # s=0
        [[0, 0], [1, -1]],  # s=1
        [[0, 1], [0, 1]],   # s=2
        [[1, 0], [1, 0]],   # s=3
        [[-1, 1], [0, 0]],  # s=4
        [[0, 0], [0, 1]],   # s=5
        [[1, 0], [0, 0]],   # s=6
    ]

    # Block-level C contributions for each t:
    # t=0: C00 += 1, C11 += 1  -> block C = [[1, 0], [0, 1]]
    # t=1: C10 += 1, C11 -= 1  -> block C = [[0, 0], [1, -1]]
    # t=2: C01 += 1, C11 += 1  -> block C = [[0, 1], [0, 1]]
    # t=3: C00 += 1, C10 += 1  -> block C = [[1, 0], [1, 0]]
    # t=4: C00 -= 1, C01 += 1  -> block C = [[-1, 1], [0, 0]]
    # t=5: C11 += 1            -> block C = [[0, 0], [0, 1]]
    # t=6: C00 += 1            -> block C = [[1, 0], [0, 0]]

    C_block = [
        [[1, 0], [0, 1]],   # t=0
        [[0, 0], [1, -1]],  # t=1
        [[0, 1], [0, 1]],   # t=2
        [[1, 0], [1, 0]],   # t=3
        [[-1, 1], [0, 0]],  # t=4
        [[0, 0], [0, 1]],   # t=5
        [[1, 0], [0, 0]],   # t=6
    ]

    # Now, for each (t, s), the total C-side is the element-wise product of
    # C_strassen[s] and C_block[t].

    # The A-side linear form for (t, s) is a dict {(i,j): coeff} for the 4x4 A matrix.
    # The B-side linear form is a dict {(i,j): coeff} for the 4x4 B matrix.

    def combine_blocks(block_dicts):
        """Combine a list of block-dicts into a single dict."""
        result = {}
        for bd in block_dicts:
            for key, val in bd.items():
                result[key] = result.get(key, 0) + val
        return result

    # Build the A-side and B-side linear forms for each (t, s).
    for t in range(7):
        A_t = A_sides[t]
        B_t = B_sides[t]
        for s in range(7):
            col = t * 7 + s

            # Determine A-side linear form based on s.
            if s == 0:
                A_form = combine_blocks([A_t[0][0], A_t[1][1]])
                B_form = combine_blocks([B_t[0][0], B_t[1][1]])
            elif s == 1:
                A_form = combine_blocks([A_t[1][0], A_t[1][1]])
                B_form = combine_blocks([B_t[0][0]])
            elif s == 2:
                A_form = combine_blocks([A_t[0][0]])
                B_form = combine_blocks([B_t[0][1], {k: -v for k, v in B_t[1][1].items()}])
            elif s == 3:
                A_form = combine_blocks([A_t[1][1]])
                B_form = combine_blocks([B_t[1][0], {k: -v for k, v in B_t[0][0].items()}])
            elif s == 4:
                A_form = combine_blocks([A_t[0][0], A_t[0][1]])
                B_form = combine_blocks([B_t[1][1]])
            elif s == 5:
                A_form = combine_blocks([A_t[1][0], {k: -v for k, v in A_t[0][0].items()}])
                B_form = combine_blocks([B_t[0][0], B_t[0][1]])
            elif s == 6:
                A_form = combine_blocks([A_t[1][1], {k: -v for k, v in A_t[0][1].items()}])
                B_form = combine_blocks([B_t[1][0], B_t[1][1]])

            # Fill U and V.
            for (i, j), coeff in A_form.items():
                if coeff != 0:
                    U[i * m + j][col] += coeff
            for (j, k), coeff in B_form.items():
                if coeff != 0:
                    V[j * p + k][col] += coeff

            # Fill W: the C-side is C_strassen[s] * C_block[t], element-wise.
            # C_strassen[s] is a 2x2 matrix giving the contribution to the
            # 2x2 result within the block product. C_block[t] gives the
            # contribution of that block product to the 4x4 result.
            # The final C contribution for each (i,k) is:
            #   sum over block (bi, bj) of C_strassen[s][bi][bj] * C_block[t][bi][bj]
            #   where (i,k) is in block (bi, bj) at position (i%2, k%2).
            #
            # Actually, C_strassen[s] gives the 2x2 result of the Strassen
            # product within the block. C_block[t] gives how that block product
            # contributes to the 4x4 C. So the total contribution to C[bi][bj]
            # (as a 2x2 block) is C_strassen[s] * C_block[t][bi][bj] (scalar mult).
            #
            # Wait, I need to be more careful. Let me reconsider.
            #
            # The block product S_t is a 2x2 matrix. It's computed as:
            #   S_t = Strassen_result(A_t, B_t)
            # where Strassen_result uses the 7 scalar products.
            #
            # The 4x4 C is:
            #   C = [[C00, C01], [C10, C11]]
            #   C00 = S1 + S4 - S5 + S7
            #   C01 = S3 + S5
            #   C10 = S2 + S4
            #   C11 = S1 - S2 + S3 + S6
            #
            # So C[bi][bj] = sum_t C_block[t][bi][bj] * S_t
            #
            # And S_t = Strassen_result(A_t, B_t), which is a 2x2 matrix.
            # The (r,c) entry of S_t is:
            #   S_t[r][c] = sum_s C_strassen[s][r][c] * (A_t linear form for s) * (B_t linear form for s)
            #
            # So C[bi][bj][r][c] = sum_t sum_s C_block[t][bi][bj] * C_strassen[s][r][c] * A_form[t,s] * B_form[t,s]
            #
            # The W matrix entry for (k, i) where i is in block (bi, bj) at (i%2, j%2)... 
            # Wait, I need to map (i, k) to the correct block and position.
            #
            # C[i][k] where i in {0,1,2,3}, k in {0,1,2,3}.
            # i belongs to block row bi = i//2, position r = i%2.
            # k belongs to block col bj = k//2, position c = k%2.
            #
            # C[i][k] = C[bi][bj][r][c] = sum_t sum_s C_block[t][bi][bj] * C_strassen[s][r][c] * A_form[t,s] * B_form[t,s]
            #
            # But in the tensor decomposition, C[i][k] = sum_j sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
            #
            # The W factor for term (t,s) at column col = t*7+s should be:
            #   W[k*n+i, col] = C_block[t][bi][bj] * C_strassen[s][r][c]
            # where bi = i//2, bj = k//2, r = i%2, c = k%2.

            for i in range(n):
                for k in range(p):
                    bi = i // 2
                    bj = k // 2
                    r = i % 2
                    c = k % 2
                    w_val = C_block[t][bi][bj] * C_strassen[s][r][c]
                    if w_val != 0:
                        W[k * n + i][col] += w_val

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
