"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 7]

# EVOLVE-BLOCK-START
def construct():
    # Hybrid decomposition: 3 Strassen blocks (rank 7 each) + 1 schoolbook block (rank 4)
    # for multiplying a 2x2 matrix by a 2x7 matrix.
    # U: (4, rank) = (n*m, rank)
    # V: (14, rank) = (m*p, rank)
    # W: (14, rank) = (p*n, rank) with W[k*n+i, r]
    n, m, p = DIMENSIONS
    total_rank = 3 * 7 + 4  # 25
    
    U = [[0] * total_rank for _ in range(n * m)]  # 4 rows
    V = [[0] * total_rank for _ in range(m * p)]  # 14 rows
    W = [[0] * total_rank for _ in range(p * n)]  # 14 rows
    
    # Strassen decomposition for 2x2 * 2x2 block
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)(B00)
    # M3 = A00(B01-B11)
    # M4 = A11(B10-B00)
    # M5 = (A00+A01)(B11)
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    #
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # U coefficients for each Strassen multiplier (rows: A00, A01, A10, A11)
    strassen_U = [
        [1, 0, 0, 1],   # M1: A00 + A11
        [0, 0, 1, 1],   # M2: A10 + A11
        [1, 0, 0, 0],   # M3: A00
        [0, 0, 0, 1],   # M4: A11
        [1, 1, 0, 0],   # M5: A00 + A01
        [1, 0, -1, 0],  # M6: A10 - A00
        [0, 1, 0, -1],  # M7: A01 - A11
    ]
    
    # V coefficients for each Strassen multiplier (rows: B00, B01, B10, B11)
    strassen_V = [
        [1, 0, 0, 1],   # M1: B00 + B11
        [1, 0, 0, 0],   # M2: B00
        [0, 1, 0, -1],  # M3: B01 - B11
        [0, 0, 1, -1],  # M4: B10 - B00
        [0, 0, 0, 1],   # M5: B11
        [1, 1, 0, 0],   # M6: B00 + B01
        [0, 0, 1, 1],   # M7: B10 + B11
    ]
    
    # W coefficients for each Strassen multiplier (rows: C00, C01, C10, C11)
    strassen_W = [
        [1, 0, 0, 1],   # M1: C00 + C11
        [0, 0, 1, -1],  # M2: C10 - C11
        [0, 1, 0, 1],   # M3: C01 + C11
        [1, 0, 1, 0],   # M4: C00 + C10
        [-1, 1, 0, 0],  # M5: -C00 + C01
        [0, 0, 0, 1],   # M6: C11
        [1, 0, 0, 0],   # M7: C00
    ]
    
    # Process 3 Strassen blocks for columns 0-1, 2-3, 4-5
    for block in range(3):
        col_offset = block * 2  # column index in B (k)
        for mult_idx in range(7):
            r = block * 7 + mult_idx
            # U: same for all blocks
            for i in range(4):
                U[i][r] = strassen_U[mult_idx][i]
            # V: rows are j*p + k, where j in {0,1}, k in {col_offset, col_offset+1}
            for j in range(2):
                for k_local in range(2):
                    k = col_offset + k_local
                    row_v = j * p + k
                    U_row = j * 2 + k_local  # local index in strassen_V
                    V[row_v][r] = strassen_V[mult_idx][U_row]
            # W: rows are k*n + i, where k in {col_offset, col_offset+1}, i in {0,1}
            for k_local in range(2):
                k = col_offset + k_local
                for i in range(2):
                    row_w = k * n + i
                    W_local_idx = k_local * 2 + i  # local index in strassen_W
                    W[row_w][r] = strassen_W[mult_idx][W_local_idx]
    
    # Process 1 schoolbook block for column 6
    # Schoolbook for 2x2 * 2x1: 4 multipliers
    # M1 = A00 * B06
    # M2 = A01 * B16
    # M3 = A10 * B06
    # M4 = A11 * B16
    # C06 = M1 + M2
    # C16 = M3 + M4
    
    schoolbook_U = [
        [1, 0, 0, 0],   # M1: A00
        [0, 1, 0, 0],   # M2: A01
        [0, 0, 1, 0],   # M3: A10
        [0, 0, 0, 1],   # M4: A11
    ]
    
    # V rows for block: j=0,k=6 -> row 6; j=1,k=6 -> row 13
    schoolbook_V = [
        [1, 0],   # M1: B06
        [0, 1],   # M2: B16
        [1, 0],   # M3: B06
        [0, 1],   # M4: B16
    ]
    
    # W rows for block: k=6,i=0 -> row 12; k=6,i=1 -> row 13
    schoolbook_W = [
        [1, 0],   # M1: C06
        [1, 0],   # M2: C06
        [0, 1],   # M3: C16
        [0, 1],   # M4: C16
    ]
    
    for mult_idx in range(4):
        r = 21 + mult_idx
        for i in range(4):
            U[i][r] = schoolbook_U[mult_idx][i]
        for j in range(2):
            row_v = j * p + 6
            V[row_v][r] = schoolbook_V[mult_idx][j]
        for i in range(2):
            row_w = 6 * n + i
            W[row_w][r] = schoolbook_W[mult_idx][i]
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
