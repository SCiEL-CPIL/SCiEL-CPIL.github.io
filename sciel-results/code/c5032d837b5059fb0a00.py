import time

DIMENSIONS = [2, 4, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Rank 52 solution:
    # Split m=4 into 2+2.
    # For each 2x2 A block, multiply by 2x7 B block.
    # 2x2x7 split into 2x2x2 (Rank 7) + 2x2x2 (Rank 7) + 2x2x3 (Rank 12).
    # Total per block: 26. Two blocks: 52.
    
    rank = 52
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    def add_strassen_222(u_offset, v_offset, w_offset):
        # u_offset: global row index of A[0,0]
        # v_offset: global row index of B[0,0]
        # w_offset: global row index of C[0,0]
        # Local indices:
        # A: i in 0..1, j in 0..1. Row i*2+j relative to u_offset? 
        # No, u_offset is the base.
        # Global A row: u_offset + i*2 + j
        # Global B row: v_offset + j*2 + k
        # Global C row: w_offset + k*2 + i
        
        col = add_strassen_222.col_counter
        # M1: (A00+A11)(B00+B11) -> C00+1, C11+1
        U[u_offset + 0][col] = 1
        U[u_offset + 3][col] = 1
        V[v_offset + 0][col] = 1
        V[v_offset + 3][col] = 1
        W[w_offset + 0][col] = 1
        W[w_offset + 3][col] = 1
        add_strassen_222.col_counter += 1
        
        col = add_strassen_222.col_counter
        # M2: (A10+A11)B00 -> C10+1, C11-1
        U[u_offset + 2][col] = 1
        U[u_offset + 3][col] = 1
        V[v_offset + 0][col] = 1
        W[w_offset + 1][col] = 1
        W[w_offset + 3][col] = -1
        add_strassen_222.col_counter += 1
        
        col = add_strassen_222.col_counter
        # M3: A00(B10-B11) -> C01+1, C11+1
        U[u_offset + 0][col] = 1
        V[v_offset + 2][col] = 1
        V[v_offset + 3][col] = -1
        W[w_offset + 2][col] = 1
        W[w_offset + 3][col] = 1
        add_strassen_222.col_counter += 1
        
        col = add_strassen_222.col_counter
        # M4: A11(B01-B00) -> C00+1, C10+1
        U[u_offset + 3][col] = 1
        V[v_offset + 1][col] = 1
        V[v_offset + 0][col] = -1
        W[w_offset + 0][col] = 1
        W[w_offset + 1][col] = 1
        add_strassen_222.col_counter += 1
        
        col = add_strassen_222.col_counter
        # M5: (A00+A01)B11 -> C00-1, C01+1
        U[u_offset + 0][col] = 1
        U[u_offset + 1][col] = 1
        V[v_offset + 3][col] = 1
        W[w_offset + 0][col] = -1
        W[w_offset + 2][col] = 1
        add_strassen_222.col_counter += 1
        
        col = add_strassen_222.col_counter
        # M6: (A10-A00)(B00+B01) -> C11+1
        U[u_offset + 2][col] = 1
        U[u_offset + 0][col] = -1
        V[v_offset + 0][col] = 1
        V[v_offset + 1][col] = 1
        W[w_offset + 3][col] = 1
        add_strassen_222.col_counter += 1
        
        col = add_strassen_222.col_counter
        # M7: (A01-A11)(B10+B11) -> C00+1
        U[u_offset + 1][col] = 1
        U[u_offset + 3][col] = -1
        V[v_offset + 2][col] = 1
        V[v_offset + 3][col] = 1
        W[w_offset + 0][col] = 1
        add_strassen_222.col_counter += 1

    def add_standard_223(u_offset, v_offset, w_offset):
        # u_offset: global row index of A[0,0]
        # v_offset: global row index of B[0,0]
        # w_offset: global row index of C[0,0]
        # A: 2x2. B: 2x3. C: 2x3.
        # Global A row: u_offset + i*2 + j
        # Global B row: v_offset + j*3 + k
        # Global C row: w_offset + k*2 + i
        
        col = add_standard_223.col_counter
        for i in range(2):
            for j in range(2):
                for k in range(3):
                    U[u_offset + i * 2 + j][col] = 1
                    V[v_offset + j * 3 + k][col] = 1
                    W[w_offset + k * 2 + i][col] = 1
                    add_standard_223.col_counter += 1

    # Initialize counters
    add_strassen_222.col_counter = 0
    add_standard_223.col_counter = 0
    
    # Block 0: A1 (cols 0,1), B1 (rows 0,1)
    # A1 base row: 0
    # B1 base row: 0
    # C base row: 0
    
    # B1 part 0: cols 0,1 (2x2x2)
    add_strassen_222(0, 0, 0)
    
    # B1 part 1: cols 2,3 (2x2x2)
    # A same. B offset: 2*2 = 4? No.
    # B1 is 2x7.
    # B1 part 1 uses B rows 0,1 and cols 2,3.
    # Global B row index: j*7 + k.
    # For j=0, k=2 -> row 2.
    # For j=1, k=2 -> row 9.
    # So v_offset for B1 part 1 is 2?
    # Wait, v_offset is the row for B[0,0] of the local block.
    # Local B[0,0] corresponds to Global B[0, 2]. Row 2.
    # Local B[1,0] corresponds to Global B[1, 2]. Row 9.
    # Difference is 7.
    # In add_strassen_222, we assume B rows are contiguous?
    # V[v_offset + j*2 + k].
    # If v_offset=2, then j=0,k=0 -> 2. j=1,k=0 -> 2+2+0=4.
    # But Global B[1,2] is row 9.
    # So we cannot use a simple offset if the stride is not 1.
    # We need to map indices explicitly or adjust the helper.
    
    # Let's rewrite helpers to take explicit index mappings or assume contiguous blocks in memory?
    # The matrices U, V, W are dense.
    # We can just pass the base offsets if the blocks are contiguous.
    # In V (size 28x52), the rows are j*p+k.
    # For B1 (j=0,1), rows are 0..6 and 7..13.
    # Part 0: k=0,1. Rows 0,1 and 7,8. Not contiguous.
    # So we must modify the helpers to accept a function or list of indices.
    
    # Let's simplify: just write the loops inline for each block part.
    
    # Reset U, V, W
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Helper to add Strassen 2x2x2
    def add_strassen_222_explicit(u_rows, v_rows, w_rows):
        # u_rows: list of 4 global U row indices for A00, A01, A10, A11
        # v_rows: list of 4 global V row indices for B00, B01, B10, B11
        # w_rows: list of 4 global W row indices for C00, C01, C10, C11
        nonlocal col
        # M1
        U[u_rows[0]][col] = 1
        U[u_rows[3]][col] = 1
        V[v_rows[0]][col] = 1
        V[v_rows[3]][col] = 1
        W[w_rows[0]][col] = 1
        W[w_rows[3]][col] = 1
        col += 1
        # M2
        U[u_rows[2]][col] = 1
        U[u_rows[3]][col] = 1
        V[v_rows[0]][col] = 1
        W[w_rows[1]][col] = 1
        W[w_rows[3]][col] = -1
        col += 1
        # M3
        U[u_rows[0]][col] = 1
        V[v_rows[2]][col] = 1
        V[v_rows[3]][col] = -1
        W[w_rows[2]][col] = 1
        W[w_rows[3]][col] = 1
        col += 1
        # M4
        U[u_rows[3]][col] = 1
        V[v_rows[1]][col] = 1
        V[v_rows[0]][col] = -1
        W[w_rows[0]][col] = 1
        W[w_rows[1]][col] = 1
        col += 1
        # M5
        U[u_rows[0]][col] = 1
        U[u_rows[1]][col] = 1
        V[v_rows[3]][col] = 1
        W[w_rows[0]][col] = -1
        W[w_rows[2]][col] = 1
        col += 1
        # M6
        U[u_rows[2]][col] = 1
        U[u_rows[0]][col] = -1
        V[v_rows[0]][col] = 1
        V[v_rows[1]][col] = 1
        W[w_rows[3]][col] = 1
        col += 1
        # M7
        U[u_rows[1]][col] = 1
        U[u_rows[3]][col] = -1
        V[v_rows[2]][col] = 1
        V[v_rows[3]][col] = 1
        W[w_rows[0]][col] = 1
        col += 1

    # Helper to add Standard 2x2x3
    def add_standard_223_explicit(u_rows, v_rows, w_rows):
        # u_rows: list of 4 global U row indices
        # v_rows: list of 6 global V row indices (B00, B01, B02, B10, B11, B12)
        # w_rows: list of 6 global W row indices (C00, C01, C02, C10, C11, C12)
        nonlocal col
        for i in range(2):
            for j in range(2):
                for k in range(3):
                    U[u_rows[i * 2 + j]][col] = 1
                    V[v_rows[j * 3 + k]][col] = 1
                    W[w_rows[k * 2 + i]][col] = 1
                    col += 1

    # Block 0: A1 (j=0,1), B1 (j=0,1)
    # A1 rows:
    # A[0,0]=0, A[0,1]=1, A[1,0]=4, A[1,1]=5
    u_rows_b0 = [0, 1, 4, 5]
    
    # B1 parts:
    # Part 0: k=0,1. B[0,0]=0, B[0,1]=1, B[1,0]=7, B[1,1]=8
    v_rows_b0_p0 = [0, 1, 7, 8]
    # C rows: C[0,0]=0, C[0,1]=2, C[1,0]=1, C[1,1]=3
    w_rows_b0 = [0, 2, 1, 3]
    
    add_strassen_222_explicit(u_rows_b0, v_rows_b0_p0, w_rows_b0)
    
    # Part 1: k=2,3. B[0,2]=2, B[0,3]=3, B[1,2]=9, B[1,3]=10
    v_rows_b0_p1 = [2, 3, 9, 10]
    # C rows for k=2,3:
    # C[0,2]=4, C[0,3]=6, C[1,2]=5, C[1,3]=7
    w_rows_b0_p1 = [4, 6, 5, 7]
    
    add_strassen_222_explicit(u_rows_b0, v_rows_b0_p1, w_rows_b0_p1)
    
    # Part 2: k=4,5,6. B[0,4]=4, B[0,5]=5, B[0,6]=6, B[1,4]=11, B[1,5]=12, B[1,6]=13
    v_rows_b0_p2 = [4, 5, 6, 11, 12, 13]
    # C rows for k=4,5,6:
    # C[0,4]=8, C[0,5]=10, C[0,6]=12, C[1,4]=9, C[1,5]=11, C[1,6]=13
    w_rows_b0_p2 = [8, 10, 12, 9, 11, 13]
    
    add_standard_223_explicit(u_rows_b0, v_rows_b0_p2, w_rows_b0_p2)
    
    # Block 1: A2 (j=2,3), B2 (j=2,3)
    # A2 rows:
    # A[0,2]=2, A[0,3]=3, A[1,2]=6, A[1,3]=7
    u_rows_b1 = [2, 3, 6, 7]
    
    # B2 parts:
    # Part 0: k=0,1. B[2,0]=14, B[2,1]=15, B[3,0]=21, B[3,1]=22
    v_rows_b1_p0 = [14, 15, 21, 22]
    # C rows same as Block 0 Part 0
    w_rows_b1_p0 = [0, 2, 1, 3]
    
    add_strassen_222_explicit(u_rows_b1, v_rows_b1_p0, w_rows_b1_p0)
    
    # Part 1: k=2,3. B[2,2]=16, B[2,3]=17, B[3,2]=23, B[3,3]=24
    v_rows_b1_p1 = [16, 17, 23, 24]
    w_rows_b1_p1 = [4, 6, 5, 7]
    
    add_strassen_222_explicit(u_rows_b1, v_rows_b1_p1, w_rows_b1_p1)
    
    # Part 2: k=4,5,6. B[2,4]=18, B[2,5]=19, B[2,6]=20, B[3,4]=25, B[3,5]=26, B[3,6]=27
    v_rows_b1_p2 = [18, 19, 20, 25, 26, 27]
    w_rows_b1_p2 = [8, 10, 12, 9, 11, 13]
    
    add_standard_223_explicit(u_rows_b1, v_rows_b1_p2, w_rows_b1_p2)
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
