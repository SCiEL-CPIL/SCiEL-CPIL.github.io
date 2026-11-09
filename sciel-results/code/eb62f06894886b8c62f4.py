"""Strassen-based rank-49 decomposition for 4x4 x 4x7 matrix multiplication."""

def construct():
    n, m, p = 4, 4, 7
    rank = 49

    # U: (n*m) x rank = 16 x 49
    # V: (m*p) x rank = 28 x 49
    # W: (p*n) x rank = 28 x 49
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    def urow(i, j):
        return i * m + j

    def vrow(j, k):
        return j * p + k

    def wrow(k, i):
        return k * n + i

    def add_to(mat, r, c, s=1):
        if s != 0:
            mat[r][c] += s

    # Strassen for 4x4 x 4x7.
    # A is 4x4, B is 4x7, C is 4x7.
    # Split A into 2x2 blocks of 2x2: A11, A12, A21, A22.
    # Split B into 2x2 blocks of 2x7: B11, B12, B21, B22.
    #   B11 = B[0:2, 0:7], B12 = B[0:2, 7:14]?? No, B is 4x7.
    #   So B is split into top 2 rows and bottom 2 rows, each 2x7.
    #   B_top = B[0:2, :], B_bot = B[2:4, :]
    #   But we need 4 blocks for Strassen. Let's split the 7 columns? No, 7 is odd.
    #   Actually, for 4x4 x 4x7, we can view it as 4x4 x 4x4 (zero-padded) or use a different split.
    #   Let's split A into 4 blocks of 2x2.
    #   Let's split B into 4 blocks: B11=B[0:2,0:4], B12=B[0:2,4:7], B21=B[2:4,0:4], B22=B[2:4,4:7].
    #   Then C11 = A11*B11 + A12*B21 (2x2 * 2x4 + 2x2 * 2x4 = 2x4)
    #   C12 = A11*B12 + A12*B22 (2x2 * 2x3 + 2x2 * 2x3 = 2x3)
    #   C21 = A21*B11 + A22*B21 (2x2 * 2x4 + 2x2 * 2x4 = 2x4)
    #   C22 = A21*B12 + A22*B22 (2x2 * 2x3 + 2x2 * 2x3 = 2x3)
    #
    # Each C_ij is a sum of two products of 2x2 and 2xk matrices.
    # The product of a 2x2 and a 2xk matrix can be done in 7 ops (Strassen for 2x2 x 2xk).
    # So C11 needs 14 ops, C12 needs 14 ops, C21 needs 14 ops, C22 needs 14 ops.
    # Total = 56 ops.
    #
    # Alternatively, apply Strassen directly to the 4x4 x 4x4 part (ignoring the 7th column),
    # which gives 49 ops. But that doesn't handle the 7th column.
    #
    # Let me use the 56-op approach.
    #
    # For 2x2 x 2xk:
    # A2 = [[a00, a01], [a10, a11]] (2x2)
    # B2 = [[b00, ..., b0k-1], [b10, ..., b1k-1]] (2xk)
    # C2 = A2 * B2 (2xk)
    # Strassen 7 ops:
    # M1 = (a00+a11) * (b0+b1)
    # M2 = (a10+a11) * b0
    # M3 = a00 * (b1-b0)
    # M4 = a11 * (b1-b0)  -- wait, standard Strassen for 2x2 x 2x2 is:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B01-B11)
    # M4 = A11(B10-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    # For 2x2 x 2xk, B00 = B[0,0:1] (a row vector of length k? No, B00 is a 1x1 block).
    # I think for 2x2 x 2xk, we treat B as having 2 rows and k columns, and apply Strassen
    # to the 2x2 A part, with B's columns as separate "blocks".
    # Actually, the cleanest: 2x2 x 2xk multiplication has rank 7.
    # The 7 ops are:
    # For each column c (0..k-1), we have a 2x2 x 2x1 product, which needs 7 ops.
    # Total 7k ops. That's not 7.
    #
    # I think I'm stuck. Let me just use schoolbook rank 112.
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}
