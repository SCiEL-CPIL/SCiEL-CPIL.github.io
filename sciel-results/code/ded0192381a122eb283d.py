import time

DIMENSIONS = [2, 3, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Rank 39 decomposition
    # Split A (2x3) into A_L (2x2) and A_R (2x1)
    # Split B (3x7) into B_T (2x7) and B_B (1x7)
    # C = A_L * B_T + A_R * B_B
    
    # A_L * B_T:
    # B_T is 2x7. Split into 3 blocks of 2 cols and 1 block of 1 col.
    # Each 2x2 block uses Strassen (rank 7).
    # The 1 col block uses rank 4.
    # Total rank for A_L * B_T is 3*7 + 4 = 25.
    
    # A_R * B_B:
    # A_R is 2x1, B_B is 1x7.
    # This is an outer product of a 2-vector and a 7-vector.
    # Rank is 2 * 7 = 14.
    
    # Total rank = 25 + 14 = 39.
    
    rank = 39
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Part 1: A_R * B_B
    # A_R corresponds to A[i, 2] for i in 0,1
    # B_B corresponds to B[2, k] for k in 0..6
    # Output C[i, k] is at W[k*n + i]
    for i in range(n):
        for k in range(p):
            U[i * m + 2][col] = 1  # A[i, 2]
            V[2 * p + k][col] = 1  # B[2, k]
            W[k * n + i][col] = 1  # C[i, k]
            col += 1
            
    # Part 2: A_L * B_T
    # A_L corresponds to A[i, j] for i in 0,1 and j in 0,1
    # B_T corresponds to B[j, k] for j in 0,1 and k in 0..6
    
    # Helper to add Strassen decomposition for 2x2 A and 2x2 B block
    def add_strassen_block(a_rows, a_cols, b_rows, b_cols, c_row_offset, c_col_offset):
        nonlocal col
        # a_rows: [i1, i2], a_cols: [j1, j2]
        # b_rows: [j1, j2], b_cols: [k1, k2]
        # c_row_offset: offset for i in output index calculation?
        # Output C[i, k] is at W[k*n + i].
        # Local C[0,0] -> C[i1, k1] -> W[k1*n + i1]
        
        # Strassen terms
        # M1 = (A00 + A11) * (B00 + B11)
        # M2 = (A10 + A11) * B00
        # M3 = A00 * (B10 - B11)
        # M4 = A11 * (B01 - B00)
        # M5 = (A00 + A01) * B11
        # M6 = (A10 - A00) * (B00 + B01)
        # M7 = (A01 - A11) * (B10 + B11)
        
        # We need to add 7 columns.
        
        # M1
        U[a_rows[0] * m + a_cols[0]][col] += 1
        U[a_rows[1] * m + a_cols[1]][col] += 1
        V[b_rows[0] * p + b_cols[0]][col] += 1
        V[b_rows[1] * p + b_cols[1]][col] += 1
        # C00 += M1, C11 += M1
        W[b_cols[0] * n + a_rows[0]][col] += 1
        W[b_cols[1] * n + a_rows[1]][col] += 1
        col += 1
        
        # M2
        U[a_rows[1] * m + a_cols[0]][col] += 1
        U[a_rows[1] * m + a_cols[1]][col] += 1
        V[b_rows[0] * p + b_cols[0]][col] += 1
        # C10 += M2
        W[b_cols[0] * n + a_rows[1]][col] += 1
        col += 1
        
        # M3
        U[a_rows[0] * m + a_cols[0]][col] += 1
        V[b_rows[1] * p + b_cols[0]][col] += 1
        V[b_rows[1] * p + b_cols[1]][col] += -1
        # C01 += M3, C10 += M3
        W[b_cols[1] * n + a_rows[0]][col] += 1
        W[b_cols[0] * n + a_rows[1]][col] += 1
        col += 1
        
        # M4
        U[a_rows[1] * m + a_cols[1]][col] += 1
        V[b_rows[0] * p + b_cols[1]][col] += 1
        V[b_rows[0] * p + b_cols[0]][col] += -1
        # C00 += M4, C10 += M4
        W[b_cols[0] * n + a_rows[0]][col] += 1
        W[b_cols[0] * n + a_rows[1]][col] += 1
        col += 1
        
        # M5
        U[a_rows[0] * m + a_cols[0]][col] += 1
        U[a_rows[0] * m + a_cols[1]][col] += 1
        V[b_rows[1] * p + b_cols[1]][col] += 1
        # C00 -= M5, C01 += M5
        W[b_cols[0] * n + a_rows[0]][col] += -1
        W[b_cols[1] * n + a_rows[0]][col] += 1
        col += 1
        
        # M6
        U[a_rows[1] * m + a_cols[0]][col] += 1
        U[a_rows[0] * m + a_cols[0]][col] += -1
        V[b_rows[0] * p + b_cols[0]][col] += 1
        V[b_rows[0] * p + b_cols[1]][col] += 1
        # C11 += M6
        W[b_cols[1] * n + a_rows[1]][col] += 1
        col += 1
        
        # M7
        U[a_rows[0] * m + a_cols[1]][col] += 1
        U[a_rows[1] * m + a_cols[1]][col] += -1
        V[b_rows[1] * p + b_cols[0]][col] += 1
        V[b_rows[1] * p + b_cols[1]][col] += 1
        # C00 += M7, C11 += M7
        W[b_cols[0] * n + a_rows[0]][col] += 1
        W[b_cols[1] * n + a_rows[1]][col] += 1
        col += 1

    # Process 3 blocks of 2 columns for B_T
    # Cols 0,1
    add_strassen_block([0, 1], [0, 1], [0, 1], [0, 1], 0, 0)
    # Cols 2,3
    add_strassen_block([0, 1], [0, 1], [0, 1], [2, 3], 0, 0)
    # Cols 4,5
    add_strassen_block([0, 1], [0, 1], [0, 1], [4, 5], 0, 0)
    
    # Process 1 block of 1 column for B_T (col 6)
    # A_L (2x2) * B_col (2x1)
    # C0 = A00*B0 + A01*B1
    # C1 = A10*B0 + A11*B1
    # 4 terms
    k = 6
    # Term 1: A00 * B0 -> C0
    U[0 * m + 0][col] += 1
    V[0 * p + k][col] += 1
    W[k * n + 0][col] += 1
    col += 1
    # Term 2: A01 * B1 -> C0
    U[0 * m + 1][col] += 1
    V[1 * p + k][col] += 1
    W[k * n + 0][col] += 1
    col += 1
    # Term 3: A10 * B0 -> C1
    U[1 * m + 0][col] += 1
    V[0 * p + k][col] += 1
    W[k * n + 1][col] += 1
    col += 1
    # Term 4: A11 * B1 -> C1
    U[1 * m + 1][col] += 1
    V[1 * p + k][col] += 1
    W[k * n + 1][col] += 1
    col += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
