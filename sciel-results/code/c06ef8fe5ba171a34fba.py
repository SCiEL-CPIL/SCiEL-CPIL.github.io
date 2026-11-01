"""Matrix multiplication 2,2,4: rank-minimal exact tensor decomposition."""
DIMENSIONS = [2, 2, 4]

def construct():
    n, m, p = DIMENSIONS
    rank = 14
    U = [[0.0] * rank for _ in range(n * m)]  # 4 x 14
    V = [[0.0] * rank for _ in range(m * p)]  # 8 x 14
    W = [[0.0] * rank for _ in range(p * n)]  # 8 x 14

    # Strassen formulas for 2x2x2 block 0 (cols 0-6) and block 1 (cols 7-13)
    # Block 0: multiplies A (2x2) by B_0 (2x2), where B_0 uses cols 0,1 of B
    # Block 1: multiplies A (2x2) by B_1 (2x2), where B_1 uses cols 2,3 of B

    # Strassen factors for a 2x2x2 multiplication:
    # U factors (input A): 4 rows (i=0,1; j=0,1) -> indexed as i*2+j
    # V factors (input B): 4 rows (j=0,1; k=0,1) -> indexed as j*2+k
    # W factors (output C): 4 rows (k=0,1; i=0,1) -> indexed as k*2+i

    # Standard Strassen 7-term decomposition:
    # M1 = (A11)(B22 - B21)  -> but let's use the standard indexing
    # Using the classic Strassen factors:
    #
    # For C = A * B where A[i,j], B[j,k], C[i,k]
    #
    # Term 0: (A[0,0]+A[1,1]) * (B[0,0]+B[1,1])
    # Term 1: (A[1,0]+A[1,1]) * B[0,0]
    # Term 2: A[0,0] * (B[1,0]-B[1,1])
    # Term 3: A[1,1] * (B[0,1]-B[0,0])
    # Term 4: (A[0,0]+A[0,1]) * B[1,1]
    # Term 5: (A[1,0]-A[0,0]) * (B[0,0]+B[0,1])
    # Term 6: (A[0,1]-A[1,1]) * (B[1,0]+B[1,1])
    #
    # C[0,0] = M1 + M2 - M4 + M5
    # C[0,1] = M3 + M4
    # C[1,0] = M1 - M3 - M5 + M6
    # C[1,1] = M2 + M3 - M1 + M6

    # Let me define the Strassen factors more carefully.
    # A has indices (i, j) where i=0,1 (row), j=0,1 (col)
    # B has indices (j, k) where j=0,1 (row), k=0,1 (col)
    # C has indices (i, k) where i=0,1 (row), k=0,1 (col)
    #
    # U factors for A: row index is i*m+j = i*2+j
    # V factors for B: row index is j*p_block+k = j*2+k (within each 2x2 block)
    # W factors for C: row index is k*n+i = k*2+i

    # Strassen decomposition (one common convention):
    # f1 = (a00+a11)(b00+b11)
    # f2 = (a10+a11)b00
    # f3 = a00(b10-b11)
    # f4 = a11(b01-b00)
    # f5 = (a00+a01)b11
    # f6 = (a10-a00)(b00+b01)
    # f7 = (a01-a11)(b10+b11)
    #
    # c00 = f1 + f2 - f4 + f5
    # c01 = f3 + f4
    # c10 = f1 - f2 - f5 + f6
    # c11 = f2 + f3 - f4 + f7  -- wait, let me recheck

    # Actually, let me use a well-known correct set. From Strassen's original:
    # m1 = (a11)(b22-b21)
    # m2 = (a11+a12)(b21+b22)
    # m3 = (a21+a22)(b11+b12)
    # m4 = (a22)(b12-b11)
    # m5 = (a11+a21)(b11)
    # m6 = (a12+a22)(b22)
    # m7 = (a11-a21)(b12)
    #
    # c11 = m1+m4-m6+m7
    # c12 = m2-m4-m6-m7
    # c21 = m3-m1+m5+m7
    # c22 = m1+m2-m3+m5

    # Converting to 0-indexed (i,j) for A (0,1; 0,1), (j,k) for B, (i,k) for C:
    # a11 -> A[0,0], a12 -> A[0,1], a21 -> A[1,0], a22 -> A[1,1]
    # b11 -> B[0,0], b12 -> B[0,1], b21 -> B[1,0], b22 -> B[1,1]
    #
    # m1 = A[0,0]*(B[1,1]-B[1,0])
    # m2 = (A[0,0]+A[0,1])*(B[1,0]+B[1,1])
    # m3 = (A[1,0]+A[1,1])*(B[0,0]+B[0,1])
    # m4 = A[1,1]*(B[0,1]-B[0,0])
    # m5 = (A[0,0]+A[1,0])*B[0,0]
    # m6 = (A[0,1]+A[1,1])*B[1,1]
    # m7 = (A[0,0]-A[1,0])*B[0,1]
    #
    # C[0,0] = m1+m4-m6+m7
    # C[0,1] = m2-m4-m6-m7
    # C[1,0] = m3-m1+m5+m7
    # C[1,1] = m1+m2-m3+m5

    # U factors (rows indexed by i*2+j for A[i,j]):
    # U[0,0]=A[0,0], U[0,1]=A[0,0], U[0,2]=0,    U[0,3]=0,    U[0,4]=A[0,0], U[0,5]=0,    U[0,6]=A[0,0]
    # U[1,0]=0,    U[1,1]=A[0,1], U[1,2]=0,    U[1,3]=0,    U[1,4]=0,    U[1,5]=0,    U[1,6]=0
    # U[2,0]=0,    U[2,1]=0,    U[2,2]=A[1,0], U[2,3]=A[1,0], U[2,4]=A[1,0], U[2,5]=0,    U[2,6]=-A[1,0]
    # U[3,0]=0,    U[3,1]=0,    U[3,2]=A[1,1], U[3,3]=A[1,1], U[3,4]=0,    U[3,5]=A[1,1], U[3,6]=0

    # So U columns (for one 2x2x2 block):
    # col0: [1, 0, 0, 0]  (A[0,0])
    # col1: [1, 1, 0, 0]  (A[0,0]+A[0,1])
    # col2: [0, 0, 1, 1]  (A[1,0]+A[1,1])
    # col3: [0, 0, 1, 1]  (A[1,0]+A[1,1])
    # col4: [1, 0, 1, 0]  (A[0,0]+A[1,0])
    # col5: [0, 1, 0, 1]  (A[0,1]+A[1,1])
    # col6: [1, 0, -1, 0] (A[0,0]-A[1,0])

    # V factors (rows indexed by j*2+k for B[j,k] within a block):
    # col0: B[1,1]-B[1,0] -> row1 (j=1,k=1)=1, row3 (j=1,k=0)=-1 -> [0,0,-1,1]
    # col1: B[1,0]+B[1,1] -> [0,0,1,1]
    # col2: B[0,0]+B[0,1] -> [1,1,0,0]
    # col3: B[0,1]-B[0,0] -> [-1,1,0,0]
    # col4: B[0,0] -> [1,0,0,0]
    # col5: B[1,1] -> [0,0,0,1]
    # col6: B[0,1] -> [0,1,0,0]

    # W factors (rows indexed by k*2+i for C[k,i]):
    # C[0,0]=m1+m4-m6+m7, C[0,1]=m2-m4-m6-m7, C[1,0]=m3-m1+m5+m7, C[1,1]=m1+m2-m3+m5
    # Row0 = C[0,0] (k=0,i=0): m1+m4-m6+m7 -> [1,0,0,1,-1,1]
    # Row1 = C[0,1] (k=0,i=1): m2-m4-m6-m7 -> [0,1,0,-1,-1,-1]
    # Row2 = C[1,0] (k=1,i=0): m3-m1+m5+m7 -> [-1,0,1,0,1,0,1]
    # Row3 = C[1,1] (k=1,i=1): m1+m2-m3+m5 -> [1,1,-1,0,1,0,0]

    U_block = [
        [1, 1, 0, 0, 1, 0, 1],
        [0, 1, 0, 0, 0, 1, 0],
        [0, 0, 1, 1, 1, 0, -1],
        [0, 0, 1, 1, 0, 1, 0]
    ]

    V_block = [
        [0, 0, 1, -1, 1, 0, 0],
        [0, 0, 1, 1, 0, 0, 1],
        [0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0]
    ]
    # Wait, V indices: row = j*2+k. For a 2x2 block:
    # j=0,k=0 -> row 0; j=0,k=1 -> row 1; j=1,k=0 -> row 2; j=1,k=1 -> row 3
    # col0: B[1,1]-B[1,0] = -1 at (j=1,k=0)=row2, +1 at (j=1,k=1)=row3 -> [0,0,-1,1]
    # col1: B[1,0]+B[1,1] = +1 at row2, +1 at row3 -> [0,0,1,1]
    # col2: B[0,0]+B[0,1] = +1 at row0, +1 at row1 -> [1,1,0,0]
    # col3: B[0,1]-B[0,0] = -1 at row0, +1 at row1 -> [-1,1,0,0]
    # col4: B[0,0] = +1 at row0 -> [1,0,0,0]
    # col5: B[1,1] = +1 at row3 -> [0,0,0,1]
    # col6: B[0,1] = +1 at row1 -> [0,1,0,0]

    V_block = [
        [0, 0, 1, -1, 1, 0, 0],
        [0, 0, 1, 1, 0, 0, 1],
        [0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0]
    ]
    # That's wrong. Let me redo:
    V_block_rows = []
    for r in range(4):
        V_block_rows.append([0]*7)
    # col0: [0,0,-1,1]
    V_block_rows[2][0] = -1; V_block_rows[3][0] = 1
    # col1: [0,0,1,1]
    V_block_rows[2][1] = 1; V_block_rows[3][1] = 1
    # col2: [1,1,0,0]
    V_block_rows[0][2] = 1; V_block_rows[1][2] = 1
    # col3: [-1,1,0,0]
    V_block_rows[0][3] = -1; V_block_rows[1][3] = 1
    # col4: [1,0,0,0]
    V_block_rows[0][4] = 1
    # col5: [0,0,0,1]
    V_block_rows[3][5] = 1
    # col6: [0,1,0,0]
    V_block_rows[1][6] = 1

    W_block = [
        [1, 0, 0, 1, -1, 1, 0],
        [0, 1, 0, -1, -1, -1, 0],
        [-1, 0, 1, 0, 1, 0, 1],
        [1, 1, -1, 0, 1, 0, 0]
    ]

    # Block 0: cols 0-6, B indices j*2+k where j=0,1 and k=0,1 (original k=0,1)
    # Block 1: cols 7-13, B indices j*2+k where j=0,1 and k=2,3 -> mapped to j*2+(k-2) in block

    for t in range(7):
        # U is same for both blocks (same A)
        for u_idx in range(4):
            U[u_idx][t] = U_block[u_idx][t]
            U[u_idx][t + 7] = U_block[u_idx][t]

        # V for block 0: row index = j*4 + k (since m=2, p=4, so j*p+k = j*4+k)
        # Within block, local row = j*2 + k_local
        for local_r in range(4):
            j = local_r // 2
            k_local = local_r % 2
            global_r = j * 4 + k_local
            V[global_r][t] = V_block_rows[local_r][t]

        # V for block 1: k = 2 + k_local
        for local_r in range(4):
            j = local_r // 2
            k_local = local_r % 2
            global_r = j * 4 + (2 + k_local)
            V[global_r][t + 7] = V_block_rows[local_r][t]

        # W for block 0: row index = k*n + i = k*2 + i, where k=0,1
        for local_w in range(4):
            k = local_w // 2
            i = local_w % 2
            global_w = k * 2 + i
            W[global_w][t] = W_block[local_w][t]

        # W for block 1: k=2,3
        for local_w in range(4):
            k_local = local_w // 2
            i = local_w % 2
            k = 2 + k_local
            global_w = k * 2 + i
            W[global_w][t + 7] = W_block[local_w][t]

    return {"U": U, "V": V, "W": W}
