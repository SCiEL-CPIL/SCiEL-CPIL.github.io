"""Matrix multiplication 2,4,5 – rank-minimal exact decomposition."""
DIMENSIONS = [2, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 2, 4, 5

    # We'll build the decomposition by partitioning the 4x5 matrix B into blocks:
    # B1 = B[:, 0:2]  (4x2)
    # B2 = B[:, 2:4]  (4x2)
    # B3 = B[:, 4:5]  (4x1)
    #
    # C = A * B = A1*B1 + A2*B2 + A3*B3
    # where A1 = A (2x4) used with B1, etc. Actually A is the same for all.
    #
    # For each block of B (4 x q), we compute A (2x4) times that block (4 x q).
    # We can use Strassen for 2x2 sub-blocks where applicable.

    # Let's use a hybrid: split A into two 2x2 blocks vertically:
    # A_top = A[0:2, :] but A is 2x4, so we split columns:
    # A_left = A[:, 0:2] (2x2), A_right = A[:, 2:4] (2x2)
    #
    # For B1 (4x2): split into B1_top = B1[0:2, :] (2x2), B1_bot = B1[2:4, :] (2x2)
    # C1 = A_left * B1_top + A_right * B1_bot
    # Each 2x2*2x2 product can use Strassen (rank 7) instead of schoolbook (rank 8).
    # So C1 costs 7 + 7 = 14.
    #
    # Similarly C2 costs 14.
    #
    # For B3 (4x1): split into B3_top = B3[0:2, :] (2x1), B3_bot = B3[2:4, :] (2x1)
    # C3 = A_left * B3_top + A_right * B3_bot
    # Each 2x2 * 2x1: schoolbook rank = 2*2*1 = 4 per, total 8.
    # (No standard Strassen for non-square; use schoolbook.)

    # Total rank: 14 + 14 + 8 = 36.

    rank = 36
    U = [[0] * rank for _ in range(n * m)]   # 8 x 36
    V = [[0] * rank for _ in range(m * p)]   # 20 x 36
    W = [[0] * rank for _ in range(p * n)]   # 10 x 36

    def idxU(i, j):
        return i * m + j

    def idxV(j, k):
        return j * p + k

    def idxW(k, i):
        return k * n + i

    col = 0

    def strassen_2x2(a_idx, b_idx, c_idx):
        """
        Strassen decomposition for C = A * B where A, B, C are 2x2.
        a_idx: list of 4 U-indices for A[00],A[01],A[10],A[11]
        b_idx: list of 4 V-indices for B[00],B[01],B[10],B[11]
        c_idx: list of 4 W-indices for C[00],C[01],C[10],C[11]
        Returns number of terms added.
        """
        global col
        A00, A01, A10, A11 = a_idx
        B00, B01, B10, B11 = b_idx
        C00, C01, C10, C11 = c_idx

        # M1 = (A00 + A11)(B00 + B11)
        U[A00][col] = 1; U[A11][col] = 1
        V[B00][col] = 1; V[B11][col] = 1
        W[C00][col] = 1; W[C10][col] = 1; W[C01][col] = -1; W[C11][col] = -1
        col += 1

        # M2 = (A10 + A11)B00
        U[A10][col] = 1; U[A11][col] = 1
        V[B00][col] = 1
        W[C10][col] = 1; W[C11][col] = 1
        col += 1

        # M3 = A00(B01 - B11)
        U[A00][col] = 1
        V[B01][col] = 1; V[B11][col] = -1
        W[C00][col] = 1; W[C01][col] = 1
        col += 1

        # M4 = A11(B10 - B00)
        U[A11][col] = 1
        V[B10][col] = 1; V[B00][col] = -1
        W[C10][col] = 1; W[C11][col] = 1
        col += 1

        # M5 = (A00 + A01)B11
        U[A00][col] = 1; U[A01][col] = 1
        V[B11][col] = 1
        W[C00][col] = 1; W[C01][col] = 1
        col += 1

        # M6 = (A10 - A00)(B00 + B01)
        U[A10][col] = 1; U[A00][col] = -1
        V[B00][col] = 1; V[B01][col] = 1
        W[C01][col] = 1; W[C11][col] = -1
        col += 1

        # M7 = (A01 - A11)(B10 + B11)
        U[A01][col] = 1; U[A11][col] = -1
        V[B10][col] = 1; V[B11][col] = 1
        W[C10][col] = -1; W[C11][col] = 1
        col += 1

        return 7

    # Block 1: C1 = A * B1, B1 is cols 0,1 of B
    # A_left = A[:, 0:2]: U indices for i=0,1; j=0,1
    #   A00 = U[0], A01 = U[1], A10 = U[4], A11 = U[5]
    # B1_top = B[0:2, 0:2]: V indices for j=0,1; k=0,1
    #   B00 = V[0], B01 = V[1], B10 = V[5], B11 = V[6]
    # C1_top portion: C[0:2, 0:2]: W indices for k=0,1; i=0,1
    #   C00 = W[0], C01 = W[1], C10 = W[2], C11 = W[3]
    # B1_bot = B[2:4, 0:2]: V indices for j=2,3; k=0,1
    #   B00 = V[10], B01 = V[11], B10 = V[15], B11 = V[16]
    # C1_bot portion: C[2:4, 0:2]: W indices for k=2,3; i=0,1
    #   C00 = W[4], C01 = W[5], C10 = W[6], C11 = W[7]

    # A_right = A[:, 2:4]: U indices for i=0,1; j=2,3
    #   A00 = U[2], A01 = U[3], A10 = U[6], A11 = U[7]

    # First Strassen: A_left * B1_top -> contributes to C rows 0,1 cols 0,1
    strassen_2x2(
        [idxU(0,0), idxU(0,1), idxU(1,0), idxU(1,1)],  # A_left
        [idxV(0,0), idxV(0,1), idxV(1,0), idxV(1,1)],  # B1_top
        [idxW(0,0), idxW(0,1), idxW(1,0), idxW(1,1)]   # C top-left
    )

    # Second Strassen: A_right * B1_bot -> contributes to C rows 0,1 cols 0,1
    strassen_2x2(
        [idxU(0,2), idxU(0,3), idxU(1,2), idxU(1,3)],  # A_right
        [idxV(2,0), idxV(2,1), idxV(3,0), idxV(3,1)],  # B1_bot
        [idxW(0,0), idxW(0,1), idxW(1,0), idxW(1,1)]   # C top-left (add)
    )

    # Block 2: C2 = A * B2, B2 is cols 2,3 of B
    # B2_top = B[0:2, 2:4]: V indices for j=0,1; k=2,3
    #   B00 = V[2], B01 = V[3], B10 = V[7], B11 = V[8]
    # C2_top portion: C[0:2, 2:4]: W indices for k=2,3; i=0,1
    #   C00 = W[4], C01 = W[5], C10 = W[6], C11 = W[7]
    # Wait, that conflicts with above. Let me re-index.
    #
    # W index: k*n + i where k is col of C (0..4), i is row of C (0..1)
    # C top-left (rows 0,1 cols 0,1): k=0,1; i=0,1 -> W[0],W[1],W[2],W[3]
    # C top-mid (rows 0,1 cols 2,3): k=2,3; i=0,1 -> W[4],W[5],W[6],W[7]
    # C top-right (rows 0,1 col 4): k=4; i=0,1 -> W[8],W[9]

    # Redo: The first two Strassen above wrote to W[0..3] which is correct for cols 0,1.
    # Now for cols 2,3:
    strassen_2x2(
        [idxU(0,0), idxU(0,1), idxU(1,0), idxU(1,1)],  # A_left
        [idxV(0,2), idxV(0,3), idxV(1,2), idxV(1,3)],  # B2_top
        [idxW(2,0), idxW(2,1), idxW(3,0), idxW(3,1)]   # C rows 0,1 cols 2,3
    )

    strassen_2x2(
        [idxU(0,2), idxU(0,3), idxU(1,2), idxU(1,3)],  # A_right
        [idxV(2,2), idxV(2,3), idxV(3,2), idxV(3,3)],  # B2_bot
        [idxW(2,0), idxW(2,1), idxW(3,0), idxW(3,1)]   # C rows 0,1 cols 2,3 (add)
    )

    # Block 3: C3 = A * B3, B3 is col 4 of B (4x1)
    # Schoolbook: 4 terms for A_left * B3_top, 4 terms for A_right * B3_bot
    # B3_top = B[0:2, 4]: V indices for j=0,1; k=4 -> V[4], V[9]
    # B3_bot = B[2:4, 4]: V indices for j=2,3; k=4 -> V[14], V[19]
    # C3: C[0:2, 4]: W indices for k=4; i=0,1 -> W[8], W[9]

    # A_left * B3_top: 2x2 * 2x1, rank 4
    # For i in 0,1; j in 0,1: U[i*m+j], V[j*p+4], W[4*n+i]
    for i in range(2):
        for j in range(2):
            U[idxU(i,j)][col] = 1
            V[idxV(j,4)][col] = 1
            W[idxW(4,i)][col] = 1
            col += 1

    # A_right * B3_bot: 2x2 * 2x1, rank 4
    for i in range(2):
        for j in range(2,4):
            U[idxU(i,j)][col] = 1
            V[idxV(j,4)][col] = 1
            W[idxW(4,i)][col] = 1
            col += 1

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
