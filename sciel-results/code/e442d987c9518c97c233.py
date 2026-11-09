"""Matrix multiplication 3x5 x 5x6 -> 3x6, rank-minimal attempt.

Strategy: treat the product as a 2x2 block multiplication over the OUTPUT:
  row blocks: R0 = rows {0,1} (size 2), R1 = row {2} (size 1)
  col blocks: C0 = cols {0,1,2} (size 3), C1 = cols {3,4,5} (size 3)

Each block is a matrix product:
  C00 (2x3) = A00 (2x5) * B00 (5x3)
  C01 (2x3) = A00 (2x5) * B01 (5x3)
  C10 (1x3) = A10 (1x5) * B00 (5x3)
  C11 (1x3) = A10 (1x5) * B01 (5x3)

We use the rank-7 Strassen scheme on these four block products, where the
"matrices" have non-square shapes but Strassen's bilinear identities hold
for arbitrary block sizes. The seven Strassen products are:

  M1 = (A10 + A00) * (B00 + B01)   # (3x5)*(5x3)
  M2 = (A10) * (B01)               # (1x5)*(5x3)
  M3 = (A00) * (B00)               # (2x5)*(5x3)
  M4 = (A10) * (B00)               # (1x5)*(5x3)
  M5 = (A00) * (B01 - B00)         # (2x5)*(5x3)
  M6 = (A10 + A00) * (B00)         # (3x5)*(5x3)
  M7 = (A10 - A2? ) ... 

Standard Strassen for 2x2:
  C00 = M1 - M5 - M4 - M6
  C01 = M3 + M5
  C10 = M2 + M4
  C11 = M1 - M2 - M3 + M7
with
  M1 = (A00 + A10)*(B00 + B01)
  M2 = (A10 - A00? ) ... let me use the standard:
  M1 = (A00 + A10)*(B00 + B01)
  M2 = (A10)*(B00 + B01)
  M3 = (A00)*(B01)
  M4 = (A10)*(B00)
  M5 = (A00)*(B00 - B01)
  M6 = (A00 + A10)*(B00)
  M7 = (A10 - A00)*(B01)
  C00 = M1 - M5 - M4 - M6
  C01 = M3 + M5
  C10 = M2 + M4
  C11 = M1 - M2 - M3 + M7

Each Mi is a rank-1 (in the bilinear sense) product of a linear form in A
and a linear form in B, so each contributes one rank-1 term to the tensor.
Total rank = 7.
"""

DIMENSIONS = [3, 5, 6]

def construct():
    n, m, p = DIMENSIONS  # 3,5,6
    # Output is 3x6. Block partition:
    # rows: R0 = [0,1] (2 rows), R1 = [2] (1 row)
    # cols: C0 = [0,1,2] (3 cols), C1 = [3,4,5] (3 cols)
    # A is 3x5 (n x m), B is 5x6 (m x p).
    # A00 = A rows 0,1 (2x5); A10 = A row 2 (1x5)
    # B00 = B cols 0,1,2 (5x3); B01 = B cols 3,4,5 (5x3)

    rank = 7
    U = [[0] * rank for _ in range(n * m)]   # 15 x 7
    V = [[0] * rank for _ in range(m * p)]   # 30 x 7
    W = [[0] * rank for _ in range(p * n)]   # 18 x 7

    # Strassen terms. Each term t corresponds to (alpha * A) * (beta * B)
    # where alpha is a linear combination of A entries, beta of B entries.
    # We represent each term by:
    #   a_coef[i][j] for A (n x m)  -> U[i*m + j, t]
    #   b_coef[j][k] for B (m x p)  -> V[j*p + k, t]
    #   c_coef[i][k] for C (n x p)  -> W[k*n + i, t]
    # such that C[i,k] += c_coef * (sum a_coef*A) * (sum b_coef*B)
    # and C[i,k] = sum_t W[k*n+i, t] * U[i*m+j,t]*V[j*p+k,t] (summed over j)

    # For a term with A-linear-form Aform (n x m matrix) and B-linear-form Bform (m x p),
    # the product (Aform * A) is NOT a scalar; it's a matrix. Wait - Strassen multiplies
    # blocks, not scalars. So each "Mi" is itself a matrix product, not a scalar.
    #
    # This means each Strassen term Mi = (LinA) * (LinB) is a full matrix product,
    # i.e. it is itself a rank-(m) term in general? No.
    #
    # Reconsider: Mi = Aform * Bform where Aform is (r x m) and Bform is (m x c).
    # Mi is an (r x c) matrix. To express Mi as a sum of rank-1 tensor terms,
    # we need Mi[i,k] = sum_j Aform[i,j]*Bform[j,k], which is m rank-1 terms.
    # So each Strassen term contributes m = 5 rank-1 terms, total 7*5 = 35.
    # That's worse than schoolbook (90)? No, 35 < 90. Good improvement!

    # So rank = 7 * m = 35.
    rank = 7 * m  # 35
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    # Define the 7 Strassen A-forms (each n x m) and B-forms (each m x p),
    # and the C contribution coefficients (each n x p, with +1/-1 placement
    # into the appropriate output block).
    #
    # A00 = A[0:2, :]; A10 = A[2:3, :]
    # B00 = B[:, 0:3]; B01 = B[:, 3:6]
    #
    # A-forms (n x m = 3 x 5):
    def Aform(coeffs):
        # coeffs: dict mapping (rowblock, colblock) -> scalar, but we need full A.
        # Simpler: define as function of A indices. We'll build matrices directly.
        pass

    # Build A-form matrices (3 x 5) and B-form matrices (5 x 6).
    # A00 occupies rows 0,1; A10 occupies row 2.
    def make_A(r0_coeff, r1_coeff):
        # r0_coeff multiplies A00 (rows 0,1), r1_coeff multiplies A10 (row 2)
        M = [[0] * m for _ in range(n)]
        for i in range(2):
            for j in range(m):
                M[i][j] = r0_coeff
        for j in range(m):
            M[2][j] = r1_coeff
        return M

    # B00 occupies cols 0,1,2; B01 occupies cols 3,4,5.
    def make_B(c0_coeff, c1_coeff):
        # c0_coeff multiplies B00 (cols 0,1,2), c1_coeff multiplies B01 (cols 3,4,5)
        M = [[0] * p for _ in range(m)]
        for j in range(m):
            for k in range(3):
                M[j][k] = c0_coeff
            for k in range(3, 6):
                M[j][k] = c1_coeff
        return M

    # Strassen terms: (Aform, Bform, Ccoeff) where Ccoeff is a (n x p) matrix
    # giving the signed placement of this term's block product into C.
    # C = C00 (rows0,1 x cols0,1,2) + C01 (rows0,1 x cols3,4,5)
    #   + C10 (row2 x cols0,1,2) + C11 (row2 x cols3,4,5)
    # C00 = M1 - M5 - M4 - M6
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 - M3 + M7

    # M1 = (A00+A10)*(B00+B01)  -> Aform(1,1), Bform(1,1)
    # M2 = (A10)*(B00+B01)      -> Aform(0,1), Bform(1,1)
    # M3 = (A00)*(B01)          -> Aform(1,0), Bform(0,1)
    # M4 = (A10)*(B00)          -> Aform(0,1), Bform(1,0)
    # M5 = (A00)*(B00-B01)      -> Aform(1,0), Bform(1,-1)
    # M6 = (A00+A10)*(B00)      -> Aform(1,1), Bform(1,0)
    # M7 = (A10-A00)*(B01)      -> Aform(-1,1), Bform(0,1)

    M = [
        (make_A(1, 1), make_B(1, 1)),   # M1
        (make_A(0, 1), make_B(1, 1)),   # M2
        (make_A(1, 0), make_B(0, 1)),   # M3
        (make_A(0, 1), make_B(1, 0)),   # M4
        (make_A(1, 0), make_B(1, -1)),  # M5
        (make_A(1, 1), make_B(1, 0)),   # M6
        (make_A(-1, 1), make_B(0, 1)),  # M7
    ]

    # C contributions: for each Mi, which output blocks get +1/-1.
    # Represent as (rowblock, colblock, sign)
    c_contrib = [
        [(-1, 0, 1), (-1, 1, -1), (-1, 2, -1), (-1, 3, -1)],  # M1: C00(+),C01(-),C10(-),C11(-)
        [(1, 0, -1), (1, 1, 1), (1, 2, -1), (1, 3, 1)],        # M2: C00(-),C01(+),C10(-),C11(+)
        [(-1, 0, 0), (-1, 1, 0), (-1, 2, 0), (-1, 3, 0)],      # placeholder
    ]
    # Let me define cleanly: for each term t (0..6), list of (rowblock, colblock, sign)
    # rowblock 0 = rows {0,1}, rowblock 1 = row {2}
    # colblock 0 = cols {0,1,2}, colblock 1 = cols {3,4,5}
    contrib = [
        [(0, 0, +1), (0, 1, -1), (1, 0, -1), (1, 1, -1)],  # M1: C00+, C01-, C10-, C11-
        [(0, 0, -1), (0, 1, +1), (1, 0, -1), (1, 1, +1)],  # M2: C00-, C01+, C10-, C11+
        [(0, 0, 0),  (0, 1, +1), (1, 0, 0),  (1, 1, -1)],  # M3: C01+, C11-
        [(0, 0, -1), (0, 1, 0),  (1, 0, +1), (1, 1, 0)],   # M4: C00-, C10+
        [(0, 0, -1), (0, 1, +1), (1, 0, 0),  (1, 1, 0)],   # M5: C00-, C01+
        [(0, 0, -1), (0, 1, 0),  (1, 0, 0),  (1, 1, 0)],   # M6: C00-
        [(0, 0, 0),  (0, 1, 0),  (1, 0, 0),  (1, 1, +1)],  # M7: C11+
    ]

    # Wait, let me recheck Strassen formulas:
    # C00 = M1 - M5 - M4 - M6
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 - M3 + M7
    # So:
    # M1: C00 +1, C11 -1
    # M2: C10 +1, C11 -1
    # M3: C01 +1, C11 -1
    # M4: C00 -1, C10 +1
    # M5: C00 -1, C01 +1
    # M6: C00 -1
    # M7: C11 +1

    contrib = [
        [(0, 0, +1), (1, 1, -1)],   # M1
        [(1, 0, +1), (1, 1, -1)],   # M2
        [(0, 1, +1), (1, 1, -1)],   # M3
        [(0, 0, -1), (1, 0, +1)],   # M4
        [(0, 0, -1), (0, 1, +1)],   # M5
        [(0, 0, -1)],               # M6
        [(1, 1, +1)],               # M7
    ]

    # Now build U, V, W. For each term t and each column j (0..m-1), we create
    # one rank-1 term. The A-form row i contributes to U[i*m + j, col], the
    # B-form row j contributes to V[j*p + k, col] for all k, and the C
    # contribution: for output (i, k), W[k*n + i, col] = sign if (i,k) is in
    # the block.
    col = 0
    for t in range(7):
        Aform, Bform = M[t]
        for j in range(m):
            for i in range(n):
                U[i * m + j][col] = Aform[i][j]
            for k in range(p):
                V[j * p + k][col] = Bform[j][k]
            # C contributions
            for (rb, cb, sign) in contrib[t]:
                rows = [0, 1] if rb == 0 else [2]
                cols = [0, 1, 2] if cb == 0 else [3, 4, 5]
                for i in rows:
                    for k in cols:
                        W[k * n + i][col] = sign
            col += 1
    assert col == rank
    return {"U": U, "V": V, "W": W}
