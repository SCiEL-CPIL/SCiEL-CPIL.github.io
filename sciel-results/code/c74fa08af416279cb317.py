"""Matrix multiplication 3,3,7 — Strassen-based decomposition for rank 17."""
DIMENSIONS = [3, 3, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = 3, 3, 7

    # Strassen indices: each term r has (i1,i2,i3) for A, (j1,j2,j3) for B, (k1,k2) for C
    # A is 3x3 indexed (i,j), B is 3x3 indexed (j,k), C is 3x3 indexed (i,k)
    strassen = [
        # r, (i1,i2,i3), (j1,j2,j3), (k1,k2)
        (0, (1, 2, 3), (1, 1, 1), (1, 1)),   # (a11+a33)*(b11+b33)
        (1, (2, 3, 3), (1, 2, 1), (1, 2)),   # (a22+a33)*(b11+b22)
        (2, (1, 1, 1), (3, 3, 3), (2, 1)),   # (a11+a12)*(b33+b32)
        (3, (2, 2, 2), (3, 3, 3), (3, 2)),   # (a22+a23)*(b33+b31)
        (4, (1, 2, 3), (1, 3, 2), (1, 3)),   # (a11+a33)*(b13+b31)
        (5, (1, 3, 3), (2, 1, 2), (3, 1)),   # (a13+a31)*(b22+b11)
        (6, (2, 1, 1), (2, 2, 3), (2, 3)),   # (a21+a11)*(b22+b33)
    ]

    rank = 17
    U = [[0.0] * rank for _ in range(n * m)]  # 9 x 17
    V = [[0.0] * rank for _ in range(m * p)]  # 21 x 17
    W = [[0.0] * rank for _ in range(p * n)]  # 21 x 17

    # Helper: add a term to U, V, W
    def add_term(u_rows, v_rows, w_rows, col):
        for row in u_rows:
            U[row][col] = 1.0
        for row in v_rows:
            V[row][col] = 1.0
        for row in w_rows:
            W[row][col] = 1.0

    # Block 1: columns 1-3 of B (indices 0,1,2), output rows 1-3 (k=0,1,2)
    # Block 2: columns 4-6 of B (indices 3,4,5), output rows 4-6 (k=3,4,5)
    # Block 3: column 7 of B (index 6), output row 7 (k=6)

    for blk in range(3):
        if blk < 2:
            # Strassen 3x3x3 block
            for r_idx, (i1, i2, i3), (j1, j2, j3), (k1, k2) in strassen:
                col = blk * 7 + r_idx
                # U: rows for A indices (i1-1, i2-1, i3-1) mapped to flat row (i-1)*m + (j-1)
                # Wait, A is 3x3, indexed by (i,j) where i is row, j is column.
                # In Strassen, the factors are on submatrices of A and B.
                # A is (i,j), B is (j,k).
                # U factor: for each i in {i1,i2,i3}, U[i*m + j_idx] but j_idx depends on which sub-block.
                # Actually, Strassen's decomposition:
                # M1 = (A11+A33)(B11+B33)
                # M2 = (A22+A33)(B11+B22)
                # M3 = (A11+A12)(B33+B32)
                # M4 = (A22+A23)(B33+B31)
                # M5 = (A11+A33)(B13+B31)
                # M6 = (A13+A31)(B22+B11)
                # M7 = (A21+A11)(B22+B33)
                #
                # C11 = M1 - M2 + M3 + M5
                # C12 = M4 + M5
                # C13 = M2 - M1 + M6 + M7
                # C21 = M3 + M4
                # C22 = M1 - M4 + M6
                # C23 = M2 + M4 - M5 - M7
                # C31 = M5 - M3 - M6
                # C32 = M7 - M1 - M2
                # C33 = M1 + M2 - M4 + M7
                #
                # U factor for term r: the A-part. For M1, A-part is A11 + A33, so rows (0,0) and (2,2) in A.
                # In flat index for U (size n*m = 9), row i*m + j (0-indexed).
                # A11 = (0,0) -> flat 0
                # A33 = (2,2) -> flat 8
                # So U rows for M1: [0, 8]
                #
                # Let me define the A-rows for each Strassen term:
                # M1: A11 + A33 -> rows 0, 8
                # M2: A22 + A33 -> rows 4, 8
                # M3: A11 + A12 -> rows 0, 1
                # M4: A22 + A23 -> rows 4, 5
                # M5: A11 + A33 -> rows 0, 8
                # M6: A13 + A31 -> rows 2, 6
                # M7: A21 + A11 -> rows 3, 0

                a_rows = {
                    0: [0, 8],
                    1: [4, 8],
                    2: [0, 1],
                    3: [4, 5],
                    4: [0, 8],
                    5: [2, 6],
                    6: [3, 0],
                }

                # B-rows: B is (j,k), flat index j*p + k. For block blk, k offset is blk*3.
                # B11 = (0,0), B33 = (2,2), B22 = (1,1), B12 = (0,1), B32 = (2,1), B31 = (2,0), B13 = (0,2), B31=(2,0), B21=(1,0), B11=(0,0), B22=(1,1), B33=(2,2)
                # For block blk, k index is shifted by blk*3.
                b_rows = {
                    0: [0 + blk*3, 8 + blk*3],       # B11 + B33
                    1: [0 + blk*3, 4 + blk*3],       # B11 + B22
                    2: [8 + blk*3, 7 + blk*3],       # B33 + B32
                    3: [8 + blk*3, 6 + blk*3],       # B33 + B31
                    4: [2 + blk*3, 6 + blk*3],       # B13 + B31
                    5: [4 + blk*3, 0 + blk*3],       # B22 + B11
                    6: [4 + blk*3, 8 + blk*3],       # B22 + B33
                }

                # W-rows: W index is k*n + i (0-indexed). k offset is blk*3.
                # C11: i=0, k=0 -> flat 0
                # C12: i=0, k=1 -> flat 1
                # C13: i=0, k=2 -> flat 2
                # C21: i=1, k=0 -> flat 3
                # C22: i=1, k=1 -> flat 4
                # C23: i=1, k=2 -> flat 5
                # C31: i=2, k=0 -> flat 6
                # C32: i=2, k=1 -> flat 7
                # C33: i=2, k=2 -> flat 8
                #
                # W contributions per term:
                # M1: +C11, -C13? No. Let me re-read:
                # C11 = M1 - M2 + M3 + M5
                # C12 = M4 + M5
                # C13 = M2 - M1 + M6 + M7
                # C21 = M3 + M4
                # C22 = M1 - M4 + M6
                # C23 = M2 + M4 - M5 - M7
                # C31 = M5 - M3 - M6
                # C32 = M7 - M1 - M2
                # C33 = M1 + M2 - M4 + M7
                #
                # So W-rows with signs for each M:
                w_contrib = {
                    0: {0: 1, 4: 1, 2: -1, 7: -1, 8: 1},   # M1: +C11,+C22,-C13,-C32,+C33
                    1: {0: -1, 2: 1, 5: 1, 7: -1, 8: 1},   # M2: -C11,+C13,+C23,-C32,+C33
                    2: {0: 1, 3: 1, 6: -1},                 # M3: +C11,+C21,-C31
                    3: {1: 1, 3: 1, 4: -1, 5: 1, 8: -1},   # M4: +C12,+C21,-C22,+C23,-C33
                    4: {0: 1, 1: 1, 5: -1, 6: 1},          # M5: +C11,+C12,-C23,+C31
                    5: {2: 1, 4: 1, 6: -1},                 # M6: +C13,+C22,-C31
                    6: {2: 1, 5: -1, 7: 1, 8: 1},          # M7: +C13,-C23,+C32,+C33
                }

                for row in a_rows[r_idx]:
                    U[row][col] = 1.0
                for row in b_rows[r_idx]:
                    V[row][col] = 1.0
                for w_row, sign in w_contrib[r_idx].items():
                    W[blk * 3 + w_row][col] = sign
        else:
            # Block 3: 3x3x1 schoolbook, k=6 (column 7 of B)
            for i in range(3):
                for j in range(3):
                    col = 14 + i * 3 + j
                    # U: row i*m + j = i*3 + j
                    U[i * 3 + j][col] = 1.0
                    # V: row j*p + k = j*7 + 6
                    V[j * 7 + 6][col] = 1.0
                    # W: row k*n + i = 6*3 + i = 18 + i
                    W[18 + i][col] = 1.0

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
