"""Rewritten program: exact rank-36 Strassen-based decomposition for 2x4 * 4x5."""
DIMENSIONS = [2, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 2, 4, 5
    # We build a rank-36 decomposition.
    # Strategy: split the inner dimension (4) into two groups of 2.
    # Group 0: rows 0,1 of A (A[i,0], A[i,1]) and rows 0,1 of B (B[0,k], B[1,k])
    #   -> computes partial C0[i,k] = A[i,0]*B[0,k] + A[i,1]*B[1,k]
    # Group 1: rows 2,3 of A and rows 2,3 of B
    #   -> computes partial C1[i,k] = A[i,2]*B[2,k] + A[i,3]*B[3,k]
    # C[i,k] = C0[i,k] + C1[i,k]
    #
    # For each group, we solve a 2x2 x 2x5 sub-problem.
    # The schoolbook rank for 2x2 x 2x5 is 20.
    # We use a Strassen-like reduction: for each pair of output columns,
    # we can potentially save terms.
    #
    # Actually, let me use a known approach:
    # For 2x2 x 2xp, we can write it as a direct sum over k of 2x2 x 2x1 problems,
    # but that gives 5*4 = 20 terms (schoolbook per k is 4).
    #
    # Better: use the fact that 2x2 x 2x2 has rank 7 (Strassen).
    # For 2x2 x 2x5, we can split the 5 columns into 2+2+1.
    #   2x2 x 2x2: rank 7 (Strassen)
    #   2x2 x 2x2: rank 7 (Strassen)
    #   2x2 x 2x1: rank 4 (schoolbook, since 2x2 x 2x1 is just 4 multiplications)
    # Total per group: 7 + 7 + 4 = 18
    # Total: 18 + 18 = 36
    #
    # Let me construct this explicitly.
    #
    # For a 2x2 x 2x2 Strassen decomposition:
    #   A = [[a, b], [c, d]], B = [[e, f], [g, h]]
    #   C = [[ae+bg, af+bh], [ce+dg, cf+dh]]
    #   Strassen uses 7 products:
    #     M1 = (a+d)(e+h)
    #     M2 = (c+d)e
    #     M3 = a(f-h)
    #     M4 = d(g-e)
    #     M5 = (a+b)h
    #     M6 = (c-a)(e+f)
    #     M7 = (b-d)(g+h)
    #   C[0,0] = M1 + M4 - M5 + M7
    #   C[0,1] = M3 + M5
    #   C[1,0] = M2 + M4
    #   C[1,1] = M1 - M2 + M3 + M6
    #
    # For 2x2 x 2x1 (just computing a 2-vector dot product result):
    #   A = [[a, b], [c, d]], B = [[e], [g]]  (2x1)
    #   C = [[ae+bg], [ce+dg]]
    #   This is just 4 terms: ae, bg, ce, dg. No saving possible for 2x2x2x1.
    #
    # Now let me lay out the full construction.
    # U has shape (n*m, rank) = (8, 36)
    # V has shape (m*p, rank) = (20, 36)
    # W has shape (p*n, rank) = (10, 36)
    #
    # Indexing:
    #   U[i*m + j, r] -> row i (0..1), inner index j (0..3)
    #   V[j*p + k, r] -> inner index j (0..3), col k (0..4)
    #   W[k*n + i, r] -> col k (0..4), row i (0..1)  [trace convention]
    #
    # Group 0 (inner indices 0,1):
    #   Sub-group 0a (output cols 0,1): 2x2 x 2x2 Strassen, rank 7
    #   Sub-group 0b (output cols 2,3): 2x2 x 2x2 Strassen, rank 7
    #   Sub-group 0c (output col 4):    2x2 x 2x1 schoolbook, rank 4
    #
    # Group 1 (inner indices 2,3):
    #   Sub-group 1a (output cols 0,1): 2x2 x 2x2 Strassen, rank 7
    #   Sub-group 1b (output cols 2,3): 2x2 x 2x2 Strassen, rank 7
    #   Sub-group 1c (output col 4):    2x2 x 2x1 schoolbook, rank 4
    #
    # Total rank: 7+7+4+7+7+4 = 36

    rank = 36
    U = [[0] * rank for _ in range(n * m)]  # 8 x 36
    V = [[0] * rank for _ in range(m * p)]  # 20 x 36
    W = [[0] * rank for _ in range(p * n)]  # 10 x 36

    col = 0

    # Helper: add Strassen 2x2x2x2 block
    # a_rows: which A rows (i indices), but here i is fixed as the outer 2x2
    # Actually, for each group, the "2x2" A is A[i, j] for i in {0,1}, j in {j1,j2}
    # The "2x2" B is B[j, k] for j in {j1,j2}, k in {k1,k2}
    # The "2x2" C output is C[i, k] for i in {0,1}, k in {k1,k2}
    def add_strassen(j1, j2, k1, k2, start_col):
        """Add a Strassen 2x2 x 2x2 block for inner rows j1,j2 and output cols k1,k2."""
        nonlocal col
        # Strassen products:
        # M1 = (a00 + a11) * (b00 + b11)
        # M2 = (a10 + a11) * b00
        # M3 = a00 * (b01 - b11)
        # M4 = a11 * (b10 - b00)
        # M5 = (a00 + a01) * b11
        # M6 = (a10 - a00) * (b00 + b01)
        # M7 = (a01 - a11) * (b10 + b11)
        #
        # In our indexing:
        # A[i,j]: i in {0,1}, j in {j1,j2}
        #   a00 = A[0,j1], a01 = A[0,j2], a10 = A[1,j1], a11 = A[1,j2]
        # B[j,k]: j in {j1,j2}, k in {k1,k2}
        #   b00 = B[j1,k1], b01 = B[j1,k2], b10 = B[j2,k1], b11 = B[j2,k2]
        # C[i,k]: i in {0,1}, k in {k1,k2}
        #   c00 = C[0,k1], c01 = C[0,k2], c10 = C[1,k1], c11 = C[1,k2]
        #
        # U row for A[i,j]: i*m + j = i*4 + j
        # V row for B[j,k]: j*p + k = j*5 + k
        # W row for C[k,i]: k*n + i = k*2 + i

        # M1 = (A[0,j1] + A[1,j2]) * (B[j1,k1] + B[j2,k2])
        # Wait, let me re-derive. The 2x2 A matrix is:
        #   [[A[0,j1], A[0,j2]],
        #    [A[1,j1], A[1,j2]]]
        # So a00=A[0,j1], a01=A[0,j2], a10=A[1,j1], a11=A[1,j2]
        # The 2x2 B matrix is:
        #   [[B[j1,k1], B[j1,k2]],
        #    [B[j2,k1], B[j2,k2]]]
        # So b00=B[j1,k1], b01=B[j1,k2], b10=B[j2,k1], b11=B[j2,k2]

        # M1 = (a00 + a11) * (b00 + b11)
        c = col; col += 1
        U[0*m + j1][c] = 1
        U[1*m + j2][c] = 1
        V[j1*p + k1][c] = 1
        V[j2*p + k2][c] = 1
        W[k1*n + 0][c] = 1
        W[k2*n + 0][c] = 1
        W[k1*n + 1][c] = 1
        W[k2*n + 1][c] = 1

        # M2 = (a10 + a11) * b00
        c = col; col += 1
        U[1*m + j1][c] = 1
        U[1*m + j2][c] = 1
        V[j1*p + k1][c] = 1
        W[k1*n + 1][c] = 1

        # M3 = a00 * (b01 - b11)
        c = col; col += 1
        U[0*m + j1][c] = 1
        V[j1*p + k2][c] = 1
        V[j2*p + k2][c] = -1
        W[k2*n + 0][c] = 1

        # M4 = a11 * (b10 - b00)
        c = col; col += 1
        U[1*m + j2][c] = 1
        V[j2*p + k1][c] = 1
        V[j1*p + k1][c] = -1
        W[k1*n + 1][c] = 1
        W[k1*n + 0][c] = 1  # M4 contributes to C[0,0] and C[1,0]

        # Wait, I need to be more careful. Let me redo the Strassen formulas.
        # C[0,0] = M1 + M4 - M5 + M7
        # C[0,1] = M3 + M5
        # C[1,0] = M2 + M4
        # C[1,1] = M1 - M2 + M3 + M6

        # Let me redo with proper W coefficients.
        # I already added M1 with W[k1,0]=1, W[k2,0]=1, W[k1,1]=1, W[k2,1]=1
        # That's wrong. M1 contributes to C[0,0], C[0,1], C[1,0], C[1,1] with different coefficients.
        # M1 appears in: C[0,0](+1), C[1,1](+1)
        # M2 appears in: C[1,0](+1), C[1,1)(-1)
        # M3 appears in: C[0,1](+1), C[1,1](+1)
        # M4 appears in: C[0,0](+1), C[1,0](+1)
        # M5 appears in: C[0,1](+1), C[0,0)(-1)
        # M6 appears in: C[1,1](+1)
        # M7 appears in: C[0,0](+1)

        # I need to reset and redo. Let me just clear and rebuild.
        return col

    # Let me just build it directly without the helper, being very careful.
    # Reset
    col = 0
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    def add_strassen_block(j1, j2, k1, k2):
        """Add 7 Strassen terms for A[0:2, j1:j2] * B[j1:j2, k1:k2]."""
        nonlocal col
        # A submatrix rows: A[0,j1], A[0,j2], A[1,j1], A[1,j2]
        # B submatrix: B[j1,k1], B[j1,k2], B[j2,k1], B[j2,k2]
        # C submatrix: C[0,k1], C[0,k2], C[1,k1], C[1,k2]

        # M1 = (A[0,j1] + A[1,j2]) * (B[j1,k1] + B[j2,k2])
        c = col; col += 1
        U[0*m + j1][c] = 1
        U[1*m + j2][c] = 1
        V[j1*p + k1][c] = 1
        V[j2*p + k2][c] = 1
        W[k1*n + 0][c] = 1   # C[0,k1] += M1
        W[k2*n + 1][c] = 1   # C[1,k2] += M1

        # M2 = (A[1,j1] + A[1,j2]) * B[j1,k1]
        c = col; col += 1
        U[1*m + j1][c] = 1
        U[1*m + j2][c] = 1
        V[j1*p + k1][c] = 1
        W[k1*n + 1][c] = 1   # C[1,k1] += M2
        W[k2*n + 1][c] = -1  # C[1,k2] -= M2

        # M3 = A[0,j1] * (B[j1,k2] - B[j2,k2])
        c = col; col += 1
        U[0*m + j1][c] = 1
        V[j1*p + k2][c] = 1
        V[j2*p + k2][c] = -1
        W[k2*n + 0][c] = 1   # C[0,k2] += M3
        W[k2*n + 1][c] = 1   # C[1,k2] += M3

        # M4 = A[1,j2] * (B[j2,k1] - B[j1,k1])
        c = col; col += 1
        U[1*m + j2][c] = 1
        V[j2*p + k1][c] = 1
        V[j1*p + k1][c] = -1
        W[k1*n + 0][c] = 1   # C[0,k1] += M4
        W[k1*n + 1][c] = 1   # C[1,k1] += M4

        # M5 = (A[0,j1] + A[0,j2]) * B[j2,k2]
        c = col; col += 1
        U[0*m + j1][c] = 1
        U[0*m + j2][c] = 1
        V[j2*p + k2][c] = 1
        W[k1*n + 0][c] = -1  # C[0,k1] -= M5
        W[k2*n + 0][c] = 1   # C[0,k2] += M5

        # M6 = (A[1,j1] - A[0,j1]) * (B[j1,k1] + B[j1,k2])
        c = col; col += 1
        U[1*m + j1][c] = 1
        U[0*m + j1][c] = -1
        V[j1*p + k1][c] = 1
        V[j1*p + k2][c] = 1
        W[k2*n + 1][c] = 1   # C[1,k2] += M6

        # M7 = (A[0,j2] - A[1,j2]) * (B[j2,k1] + B[j2,k2])
        c = col; col += 1
        U[0*m + j2][c] = 1
        U[1*m + j2][c] = -1
        V[j2*p + k1][c] = 1
        V[j2*p + k2][c] = 1
        W[k1*n + 0][c] = 1   # C[0,k1] += M7

    def add_2x2_2x1_block(j1, j2, k1):
        """Add 4 schoolbook terms for A[0:2, j1:j2] * B[j1:j2, k1:k1+1]."""
        nonlocal col
        # C[0,k1] = A[0,j1]*B[j1,k1] + A[0,j2]*B[j2,k1]
        # C[1,k1] = A[1,j1]*B[j1,k1] + A[1,j2]*B[j2,k1]
        for i in range(2):
            for jj in range(2):
                c = col; col += 1
                U[i*m + (j1 + jj)][c] = 1
                V[(j1 + jj)*p + k1][c] = 1
                W[k1*n + i][c] = 1

    # Group 0: j1=0, j2=1
    add_strassen_block(0, 1, 0, 1)  # cols 0,1 -> 7 terms
    add_strassen_block(0, 1, 2, 3)  # cols 2,3 -> 7 terms
    add_2x2_2x1_block(0, 1, 4)      # col 4 -> 4 terms

    # Group 1: j1=2, j2=3
    add_strassen_block(2, 3, 0, 1)  # cols 0,1 -> 7 terms
    add_strassen_block(2, 3, 2, 3)  # cols 2,3 -> 7 terms
    add_2x2_2x1_block(2, 3, 4)      # col 4 -> 4 terms

    # col should be 36
    assert col == 36, f"Expected col=36, got {col}"

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
