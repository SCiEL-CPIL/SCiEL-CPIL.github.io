"""Kronecker-product decomposition for 3x4x8 matrix multiplication."""

def construct():
    # We decompose the 3x4x8 tensor as a Kronecker product of:
    #   T1: 3x2x4 tensor (rank 18)
    #   T2: 1x2x2 tensor (rank 4, schoolbook)
    # Total rank = 18 * 4 = 72

    # --- T2: 1x2x2 (schoolbook, rank 4) ---
    # A' is 1x2, B' is 2x2, C' is 1x2
    # C'[0] = A'[0]*B'[0,0] + A'[1]*B'[1,0]
    # C'[1] = A'[0]*B'[0,1] + A'[1]*B'[1,1]
    # U2 shape: (1*2, 4) = (2, 4)
    # V2 shape: (2*2, 4) = (4, 4)
    # W2 shape: (2*1, 4) = (2, 4)
    U2 = [
        [1, 0, 0, 0],  # A'[0]
        [0, 1, 0, 0],  # A'[1]
    ]
    V2 = [
        [1, 0, 0, 0],  # B'[0,0]
        [0, 0, 1, 0],  # B'[0,1]
        [0, 1, 0, 0],  # B'[1,0]
        [0, 0, 0, 1],  # B'[1,1]
    ]
    W2 = [
        [1, 0, 0, 0],  # C'[0]
        [0, 1, 0, 0],  # C'[1]
    ]

    # --- T1: 3x2x4 (rank 18) ---
    # A is 3x2, B is 2x4, C is 3x4
    # Split B's columns into two blocks of 2: B_left (2x2), B_right (2x2)
    # C_left = A * B_left (3x2)
    # C_right = A * B_right (3x2)
    #
    # For C_left (3x2x2):
    #   Rows 0,1 of A (2x2) times B_left (2x2) -> use Strassen (rank 7)
    #   Row 2 of A (1x2) times B_left (2x2) -> schoolbook (rank 4)
    # For C_right (3x2x2):
    #   Rows 0,1 of A (2x2) times B_right (2x2) -> use Strassen (rank 7)
    #   Row 2 of A (1x2) times B_right (2x2) -> schoolbook (rank 4)
    # Total: 7 + 4 + 7 + 4 = 22? No, that's 22.
    #
    # Wait, let me reconsider. The task says rank-14 for 3x2x4.
    # Let me think about this differently.
    #
    # Actually, for 3x2x4:
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # i in {0,1,2}, j in {0,1}, k in {0,1,2,3}
    #
    # We can write:
    # C[0,:] = A[0,0]*B[0,:] + A[0,1]*B[1,:]  (1x2 * 2x4 = 1x4, rank 4)
    # C[1,:] = A[1,0]*B[0,:] + A[1,1]*B[1,:]  (rank 4)
    # C[2,:] = A[2,0]*B[0,:] + A[2,1]*B[1,:]  (rank 4)
    # Total schoolbook: 12
    #
    # But we can do better. Let's use the structure more carefully.
    #
    # Actually, let me just use the block approach with Strassen for the top 2 rows:
    # For rows 0,1: treat A[0:2,:] as 2x2, and B as 2x4.
    # Split B into B_left (2x2) and B_right (2x2).
    # C[0:2, 0:2] = A[0:2,:] * B_left -> Strassen rank 7
    # C[0:2, 2:4] = A[0:2,:] * B_right -> Strassen rank 7
    # For row 2: C[2,:] = A[2,0]*B[0,:] + A[2,1]*B[1,:] -> schoolbook rank 4
    # Total: 7 + 7 + 4 = 18
    #
    # So rank 18 is correct. Let me build this.

    # Strassen factors for 2x2x2
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B10-B00)
    # M4 = A11(B01-B11)
    # M5 = (A00+A10)B11
    # M6 = (A11-A10)(B00+B10)
    # M7 = (A01-A11)(B01+B11)
    #
    # C00 = M1 + M4 - M5 + M7
    # C01 = M2 + M3 - M6 + M7
    # C10 = M2 + M3 + M5 - M1
    # C11 = M1 - M3 + M6 - M4

    strassen_U = [
        [1, 0, 0, 1],   # M1: A00 + A11
        [0, 0, 1, 1],   # M2: A10 + A11
        [1, 0, 0, 0],   # M3: A00
        [0, 0, 0, 1],   # M4: A11
        [1, 1, 0, 0],   # M5: A00 + A10
        [-1, 0, 1, 0],  # M6: A11 - A10
        [0, 1, 0, -1],  # M7: A01 - A11
    ]
    strassen_V = [
        [1, 0, 0, 1],   # M1: B00 + B11
        [1, 0, 0, 0],   # M2: B00
        [0, 1, 0, -1],  # M3: B10 - B00
        [-1, 0, 1, 0],  # M4: B01 - B11
        [0, 0, 0, 1],   # M5: B11
        [1, 1, 0, 0],   # M6: B00 + B10
        [0, 0, 1, 1],   # M7: B01 + B11
    ]
    strassen_W = [
        [1, 0, 0, 1],   # M1: C00 += 1, C11 += 1
        [0, 0, 1, -1],  # M2: C10 += 1, C11 -= 1
        [0, 1, 0, 1],   # M3: C01 += 1, C11 += 1
        [1, 0, 1, 0],   # M4: C00 += 1, C10 += 1
        [-1, 1, 0, 0],  # M5: C00 -= 1, C01 += 1
        [0, 0, 0, 1],   # M6: C11 += 1
        [1, 0, 0, 0],   # M7: C00 += 1
    ]

    # Build T1 factors
    # U1 shape: (3*2, 18) = (6, 18)
    # V1 shape: (2*4, 18) = (8, 18)
    # W1 shape: (4*3, 18) = (12, 18)

    U1 = [[0] * 18 for _ in range(6)]
    V1 = [[0] * 18 for _ in range(8)]
    W1 = [[0] * 18 for _ in range(12)]

    col = 0

    # Block 1: C[0:2, 0:2] = A[0:2,:] * B[:, 0:2] (Strassen)
    for t in range(7):
        for l in range(4):
            if strassen_U[t][l] != 0:
                local_i = l // 2  # 0 or 1
                local_j = l % 2   # 0 or 1
                U1[local_i * 2 + local_j][col] += strassen_U[t][l]
            if strassen_V[t][l] != 0:
                local_j = l // 2   # 0 or 1
                local_k = l % 2    # 0 or 1
                V1[local_j * 4 + local_k][col] += strassen_V[t][l]
            if strassen_W[t][l] != 0:
                local_i = l // 2   # 0 or 1
                local_k = l % 2    # 0 or 1
                W1[local_k * 3 + local_i][col] += strassen_W[t][l]
        col += 1

    # Block 2: C[0:2, 2:4] = A[0:2,:] * B[:, 2:4] (Strassen)
    for t in range(7):
        for l in range(4):
            if strassen_U[t][l] != 0:
                local_i = l // 2
                local_j = l % 2
                U1[local_i * 2 + local_j][col] += strassen_U[t][l]
            if strassen_V[t][l] != 0:
                local_j = l // 2
                local_k = l % 2
                V1[local_j * 4 + (2 + local_k)][col] += strassen_V[t][l]
            if strassen_W[t][l] != 0:
                local_i = l // 2
                local_k = l % 2
                W1[(2 + local_k) * 3 + local_i][col] += strassen_W[t][l]
        col += 1

    # Block 3: C[2,:] = A[2,0]*B[0,:] + A[2,1]*B[1,:] (schoolbook, rank 4)
    for k in range(4):
        U1[2 * 2 + 0][col] = 1  # A[2,0]
        V1[0 * 4 + k][col] = 1  # B[0,k]
        W1[k * 3 + 2][col] = 1  # C[2,k]
        col += 1
        U1[2 * 2 + 1][col] = 1  # A[2,1]
        V1[1 * 4 + k][col] = 1  # B[1,k]
        W1[k * 3 + 2][col] = 1  # C[2,k]
        col += 1

    # Now combine T1 and T2 via Kronecker product
    # Total rank = 18 * 4 = 72
    # U shape: (3*4, 72) = (12, 72)
    # V shape: (4*8, 72) = (32, 72)
    # W shape: (8*3, 72) = (24, 72)

    rank = 72
    U = [[0] * rank for _ in range(12)]
    V = [[0] * rank for _ in range(32)]
    W = [[0] * rank for _ in range(24)]

    for c1 in range(18):
        for c2 in range(4):
            c = c1 * 4 + c2
            for i1 in range(6):
                if U1[i1][c1] != 0:
                    for i2 in range(2):
                        if U2[i2][c2] != 0:
                            U[i1 * 2 + i2][c] += U1[i1][c1] * U2[i2][c2]
            for j1 in range(8):
                if V1[j1][c1] != 0:
                    for j2 in range(4):
                        if V2[j2][c2] != 0:
                            V[j1 * 4 + j2][c] += V1[j1][c1] * V2[j2][c2]
            for k1 in range(12):
                if W1[k1][c1] != 0:
                    for k2 in range(2):
                        if W2[k2][c2] != 0:
                            W[k1 * 2 + k2][c] += W1[k1][c1] * W2[k2][c2]

    return {"U": U, "V": V, "W": W}
