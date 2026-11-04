"""Matrix multiplication 3x4x4: rank-minimal exact tensor decomposition."""

DIMENSIONS = [3, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 3, 4, 4
    rank = 44

    # U: (n*m) x rank = 12 x 44
    # V: (m*p) x rank = 16 x 44
    # W: (p*n) x rank = 12 x 44
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    col = 0

    # Helper to add a Strassen 2x2x2 block
    # A block: rows a_r..a_r+1, cols a_c..a_c+1
    # B block: rows b_r..b_r+1, cols b_c..b_c+1
    # C block: rows c_r..c_r+1, cols c_c..c_c+1
    def add_strassen_block(a_r, a_c, b_r, b_c, c_r, c_c, col_start):
        nonlocal col
        # Strassen's 7 products:
        # M1 = (A00 + A11) * (B00 + B11)
        # M2 = (A10 + A11) * B00
        # M3 = A00 * (B01 - B11)
        # M4 = A11 * (B10 - B00)
        # M5 = (A00 + A01) * B11
        # M6 = (A10 - A00) * (B00 + B01)
        # M7 = (A01 - A11) * (B10 + B11)
        #
        # C00 = M1 + M4 - M5 + M7
        # C01 = M3 + M5
        # C10 = M2 + M4
        # C11 = M1 - M2 + M3 + M6

        # M1
        U[a_r * m + a_c][col_start] = 1
        U[(a_r + 1) * m + (a_c + 1)][col_start] = 1
        V[b_r * p + b_c][col_start] = 1
        V[(b_r + 1) * p + (b_c + 1)][col_start] = 1
        W[c_r * n + c_c][col_start] = 1
        W[(c_r + 1) * n + (c_c + 1)][col_start] = 1
        col = col_start + 1

        # M2
        U[(a_r + 1) * m + a_c][col_start] = 1
        U[(a_r + 1) * m + (a_c + 1)][col_start] = 1
        V[b_r * p + b_c][col_start] = 1
        W[(c_r + 1) * n + c_c][col_start] = 1
        col = col_start + 1

        # M3
        U[a_r * m + a_c][col_start] = 1
        V[b_r * p + (b_c + 1)][col_start] = 1
        V[(b_r + 1) * p + (b_c + 1)][col_start] = -1
        W[c_r * n + (c_c + 1)][col_start] = 1
        col = col_start + 1

        # M4
        U[(a_r + 1) * m + (a_c + 1)][col_start] = 1
        V[(b_r + 1) * p + b_c][col_start] = 1
        V[b_r * p + b_c][col_start] = -1
        W[(c_r + 1) * n + c_c][col_start] = 1
        col = col_start + 1

        # M5
        U[a_r * m + a_c][col_start] = 1
        U[a_r * m + (a_c + 1)][col_start] = 1
        V[(b_r + 1) * p + (b_c + 1)][col_start] = 1
        W[c_r * n + (c_c + 1)][col_start] = -1
        col = col_start + 1

        # M6
        U[(a_r + 1) * m + a_c][col_start] = 1
        U[a_r * m + a_c][col_start] = -1
        V[b_r * p + b_c][col_start] = 1
        V[b_r * p + (b_c + 1)][col_start] = 1
        W[(c_r + 1) * n + (c_c + 1)][col_start] = 1
        col = col_start + 1

        # M7
        U[a_r * m + (a_c + 1)][col_start] = 1
        U[(a_r + 1) * m + (a_c + 1)][col_start] = -1
        V[(b_r + 1) * p + b_c][col_start] = 1
        V[(b_r + 1) * p + (b_c + 1)][col_start] = 1
        W[c_r * n + (c_c + 1)][col_start] = 1
        col = col_start + 1

    # Helper to add a schoolbook 1x2x2 block
    # A: single row a_r, cols a_c..a_c+1
    # B: rows b_r..b_r+1, cols b_c..b_c+1
    # C: single row c_r, cols c_c..c_c+1
    def add_schoolbook_block(a_r, a_c, b_r, b_c, c_r, c_c, col_start):
        nonlocal col
        # 4 products: A[0,0]*B[0,0], A[0,0]*B[1,0], A[0,1]*B[0,1], A[0,1]*B[1,1]
        # Actually for C[c_r, c_c] = sum A[a_r, a_c + d] * B[b_r + d, b_c]
        # C[c_r, c_c] = A[a_r,a_c]*B[b_r,b_c] + A[a_r,a_c+1]*B[b_r+1,b_c]
        # C[c_r, c_c+1] = A[a_r,a_c]*B[b_r,b_c+1] + A[a_r,a_c+1]*B[b_r+1,b_c+1]

        # Product 1: A[a_r, a_c] * B[b_r, b_c] -> C[c_r, c_c]
        U[a_r * m + a_c][col_start] = 1
        V[b_r * p + b_c][col_start] = 1
        W[c_r * n + c_c][col_start] = 1
        col = col_start + 1

        # Product 2: A[a_r, a_c] * B[b_r, b_c+1] -> C[c_r, c_c+1]
        U[a_r * m + a_c][col_start] = 1
        V[b_r * p + (b_c + 1)][col_start] = 1
        W[c_r * n + (c_c + 1)][col_start] = 1
        col = col_start + 1

        # Product 3: A[a_r, a_c+1] * B[b_r+1, b_c] -> C[c_r, c_c]
        U[a_r * m + (a_c + 1)][col_start] = 1
        V[(b_r + 1) * p + b_c][col_start] = 1
        W[c_r * n + c_c][col_start] = 1
        col = col_start + 1

        # Product 4: A[a_r, a_c+1] * B[b_r+1, b_c+1] -> C[c_r, c_c+1]
        U[a_r * m + (a_c + 1)][col_start] = 1
        V[(b_r + 1) * p + (b_c + 1)][col_start] = 1
        W[c_r * n + (c_c + 1)][col_start] = 1
        col = col_start + 1

    # 4 Strassen 2x2x2 blocks for rows 0-1 of A, columns 0-3 of B
    # Block 1: A[0:2, 0:2] * B[0:2, 0:2] -> C[0:2, 0:2]
    add_strassen_block(0, 0, 0, 0, 0, 0, col)
    # Block 2: A[0:2, 2:4] * B[2:4, 0:2] -> C[0:2, 0:2]
    add_strassen_block(0, 2, 2, 0, 0, 0, col)
    # Block 3: A[0:2, 0:2] * B[0:2, 2:4] -> C[0:2, 2:4]
    add_strassen_block(0, 0, 0, 2, 0, 2, col)
    # Block 4: A[0:2, 2:4] * B[2:4, 2:4] -> C[0:2, 2:4]
    add_strassen_block(0, 2, 2, 2, 0, 2, col)

    # 4 Schoolbook 1x2x2 blocks for row 2 of A
    # Block 5: A[2, 0:2] * B[0:2, 0:2] -> C[2, 0:2]
    add_schoolbook_block(2, 0, 0, 0, 2, 0, col)
    # Block 6: A[2, 0:2] * B[0:2, 2:4] -> C[2, 2:4]
    add_schoolbook_block(2, 0, 0, 2, 2, 2, col)
    # Block 7: A[2, 2:4] * B[2:4, 0:2] -> C[2, 0:2]
    add_schoolbook_block(2, 2, 2, 0, 2, 0, col)
    # Block 8: A[2, 2:4] * B[2:4, 2:4] -> C[2, 2:4]
    add_schoolbook_block(2, 2, 2, 2, 2, 2, col)

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
