"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 4]

# EVOLVE-BLOCK-START
def construct():
    # 2x2x4 matrix multiplication decomposes as a direct sum of two 2x2x2 blocks.
    # The 2x2x2 tensor has rank 7 (Strassen), so the 2x2x4 tensor has rank 14.
    # We use Strassen's 7-term algorithm for each 2x2x2 block.
    #
    # Block 1: A[0:2,0:2] * B[0:2,0:2] -> C[0:2,0:2]
    #   A1 = A[0,0], A2 = A[0,1], A3 = A[1,0], A4 = A[1,1]
    #   B1 = B[0,0], B2 = B[0,1], B3 = B[1,0], B4 = B[1,1]
    #   C1 = C[0,0], C2 = C[0,1], C3 = C[1,0], C4 = C[1,1]
    #
    # Block 2: A[0:2,2:4] * B[2:4,0:2] -> C[2:4,0:2]
    #   A5 = A[0,2], A6 = A[0,3], A7 = A[1,2], A8 = A[1,3]
    #   B5 = B[2,0], B6 = B[2,1], B7 = B[3,0], B8 = B[3,1]
    #   C5 = C[2,0], C6 = C[2,1], C7 = C[3,0], C8 = C[3,1]
    #
    # Strassen for block 1 (indices 0-6):
    #   M1 = (A3+A4)(B1+B2)
    #   M2 = (A1+A4)B3
    #   M3 = A1(B4-B3)
    #   M4 = A4(B2-B1)
    #   M5 = (A1+A2)B4
    #   M6 = (A3-A1)(B1+B4)
    #   M7 = (A2-A4)(B3+B2)
    #   C1 = M1 + M4 - M5 + M7
    #   C2 = M3 + M5
    #   C3 = M2 + M4
    #   C4 = M1 - M2 + M3 + M6
    #
    # Strassen for block 2 (indices 7-13):
    #   M8 = (A7+A8)(B5+B6)
    #   M9 = (A5+A8)B7
    #   M10 = A5(B8-B7)
    #   M11 = A8(B6-B5)
    #   M12 = (A5+A6)B8
    #   M13 = (A7-A5)(B5+B8)
    #   M14 = (A6-A8)(B7+B6)
    #   C5 = M8 + M11 - M12 + M14
    #   C6 = M10 + M12
    #   C7 = M9 + M11
    #   C8 = M8 - M9 + M10 + M13
    #
    # U matrix: rows indexed by (i*m+j) where i in [0,1], j in [0,3]
    #   Row 0: A[0,0] = A1
    #   Row 1: A[0,1] = A2
    #   Row 2: A[1,0] = A3
    #   Row 3: A[1,1] = A4
    #   Row 4: A[0,2] = A5
    #   Row 5: A[0,3] = A6
    #   Row 6: A[1,2] = A7
    #   Row 7: A[1,3] = A8
    #
    # V matrix: rows indexed by (j*p+k) where j in [0,1], k in [0,3]
    #   Row 0: B[0,0] = B1
    #   Row 1: B[0,1] = B2
    #   Row 2: B[1,0] = B3
    #   Row 3: B[1,1] = B4
    #   Row 4: B[2,0] = B5
    #   Row 5: B[2,1] = B6
    #   Row 6: B[3,0] = B7
    #   Row 7: B[3,1] = B8
    #
    # W matrix: rows indexed by (k*n+i) where k in [0,1], i in [0,3]
    #   Row 0: C[0,0] = C1
    #   Row 1: C[0,1] = C2
    #   Row 2: C[1,0] = C3
    #   Row 3: C[1,1] = C4
    #   Row 4: C[2,0] = C5
    #   Row 5: C[2,1] = C6
    #   Row 6: C[3,0] = C7
    #   Row 7: C[3,1] = C8

    rank = 14
    # U: 8 rows (2*4), 14 cols
    U = [[0] * rank for _ in range(8)]
    # V: 8 rows (2*4), 14 cols
    V = [[0] * rank for _ in range(8)]
    # W: 8 rows (4*2), 14 cols
    W = [[0] * rank for _ in range(8)]

    # Block 1: Strassen terms 0-6
    # M1 = (A3+A4)(B1+B2)
    U[2][0] = 1  # A3
    U[3][0] = 1  # A4
    V[0][0] = 1  # B1
    V[1][0] = 1  # B2

    # M2 = (A1+A4)B3
    U[0][1] = 1  # A1
    U[3][1] = 1  # A4
    V[2][1] = 1  # B3

    # M3 = A1(B4-B3)
    U[0][2] = 1  # A1
    V[3][2] = 1  # B4
    V[2][2] = -1 # B3

    # M4 = A4(B2-B1)
    U[3][3] = 1  # A4
    V[1][3] = 1  # B2
    V[0][3] = -1 # B1

    # M5 = (A1+A2)B4
    U[0][4] = 1  # A1
    U[1][4] = 1  # A2
    V[3][4] = 1  # B4

    # M6 = (A3-A1)(B1+B4)
    U[2][5] = 1  # A3
    U[0][5] = -1 # A1
    V[0][5] = 1  # B1
    V[3][5] = 1  # B4

    # M7 = (A2-A4)(B3+B2)
    U[1][6] = 1  # A2
    U[3][6] = -1 # A4
    V[2][6] = 1  # B3
    V[1][6] = 1  # B2

    # Block 2: Strassen terms 7-13
    # M8 = (A7+A8)(B5+B6)
    U[6][7] = 1  # A7
    U[7][7] = 1  # A8
    V[4][7] = 1  # B5
    V[5][7] = 1  # B6

    # M9 = (A5+A8)B7
    U[4][8] = 1  # A5
    U[7][8] = 1  # A8
    V[6][8] = 1  # B7

    # M10 = A5(B8-B7)
    U[4][9] = 1  # A5
    V[7][9] = 1  # B8
    V[6][9] = -1 # B7

    # M11 = A8(B6-B5)
    U[7][10] = 1  # A8
    V[5][10] = 1  # B6
    V[4][10] = -1 # B5

    # M12 = (A5+A6)B8
    U[4][11] = 1  # A5
    U[5][11] = 1  # A6
    V[7][11] = 1  # B8

    # M13 = (A7-A5)(B5+B8)
    U[6][12] = 1  # A7
    U[4][12] = -1 # A5
    V[4][12] = 1  # B5
    V[7][12] = 1  # B8

    # M14 = (A6-A8)(B7+B6)
    U[5][13] = 1  # A6
    U[7][13] = -1 # A8
    V[6][13] = 1  # B7
    V[5][13] = 1  # B6

    # W matrix: output coefficients
    # C1 = M1 + M4 - M5 + M7
    W[0][0] = 1   # M1
    W[0][3] = 1   # M4
    W[0][4] = -1  # M5
    W[0][6] = 1   # M7

    # C2 = M3 + M5
    W[1][2] = 1   # M3
    W[1][4] = 1   # M5

    # C3 = M2 + M4
    W[2][1] = 1   # M2
    W[2][3] = 1   # M4

    # C4 = M1 - M2 + M3 + M6
    W[3][0] = 1   # M1
    W[3][1] = -1  # M2
    W[3][2] = 1   # M3
    W[3][5] = 1   # M6

    # C5 = M8 + M11 - M12 + M14
    W[4][7] = 1   # M8
    W[4][10] = 1  # M11
    W[4][11] = -1 # M12
    W[4][13] = 1  # M14

    # C6 = M10 + M12
    W[5][9] = 1   # M10
    W[5][11] = 1  # M12

    # C7 = M9 + M11
    W[6][8] = 1   # M9
    W[6][10] = 1  # M11

    # C8 = M8 - M9 + M10 + M13
    W[7][7] = 1   # M8
    W[7][8] = -1  # M9
    W[7][9] = 1   # M10
    W[7][12] = 1  # M13

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
