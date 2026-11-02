"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 10]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # n=2, m=3, p=10
    
    # We decompose the 2x3x10 multiplication into two parts:
    # Part 1: 2x2x10 (columns 0,1 of the 3-col matrix)
    # Part 2: 2x1x10 (column 2 of the 3-col matrix)
    
    # For 2x2x10, we use 5 Strassen blocks (each 2x2x2, rank 7) = rank 35
    # For 2x1x10, we use rank 20 (each element needs its own multiplication)
    # Total rank = 35 + 20 = 55
    
    rank = 55
    U = [[0] * rank for _ in range(n * m)]  # 6 x 55
    V = [[0] * rank for _ in range(m * p)]  # 30 x 55
    W = [[0] * rank for _ in range(p * n)]  # 20 x 55
    
    # Helper to add a 2x2x2 Strassen block
    # Strassen for C = A*B where A is 2x2, B is 2x2:
    # M1 = (A11 + A22)(B11 + B22)
    # M2 = (A21 + A22)B11
    # M3 = A11(B21 - B22)
    # M4 = A22(B12 - B11)
    # M5 = (A11 + A12)B22
    # M6 = (A21 - A11)(B11 + B12)
    # M7 = (A12 - A22)(B21 + B22)
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    def add_strassen_block(base_col, a_row_offset, b_col_offset, c_row_offset, c_col_offset):
        """
        Add a 2x2x2 Strassen block.
        a_row_offset: row offset in A (2x2 block starts at this row)
        b_col_offset: col offset in B (2x2 block starts at this col)
        c_row_offset: row offset in C (2x2 block starts at this row)
        c_col_offset: col offset in C (2x2 block starts at this col)
        """
        # A is 2x2: rows a_row_offset, a_row_offset+1; cols 0,1 (within the 2-col submatrix)
        # B is 2x2: rows 0,1 (within the 2-row submatrix); cols b_col_offset, b_col_offset+1
        # C is 2x2: rows c_row_offset, c_row_offset+1; cols c_col_offset, c_col_offset+1
        
        # In the full U, V, W:
        # U has shape (n*m, rank) = (6, 55). For the 2x2 part, we use rows corresponding to A's 2x2 block.
        # The 2x2 block of A is at rows a_row_offset*2 to a_row_offset*2+1? No.
        # Let's think carefully about indexing.
        
        # For the 2x2x10 part:
        # A is 2x2 (rows 0,1 of the original 2x3 matrix, cols 0,1)
        # B is 2x10 (rows 0,1 of the original 3x10 matrix, all 10 cols)
        # C is 2x10 (rows 0,1 of the output, all 10 cols)
        
        # U rows for A: row i*m + j where i is row of A, j is col of A
        # For the 2x2 part of A: i in {0,1}, j in {0,1}
        # So U rows: 0*3+0=0, 0*3+1=1, 1*3+0=3, 1*3+1=4
        
        # V rows for B: row j*p + k where j is row of B, k is col of B
        # For the 2x10 part of B: j in {0,1}, k in {0,...,9}
        # So V rows: 0*10+0=0, 0*10+1=1, ..., 0*10+9=9, 1*10+0=10, ..., 1*10+9=19
        
        # W rows for C: row k*n + i where k is col of C, i is row of C
        # For the 2x10 part of C: k in {0,...,9}, i in {0,1}
        # So W rows: 0*2+0=0, 0*2+1=1, 1*2+0=2, 1*2+1=3, ..., 9*2+0=18, 9*2+1=19
        
        # In the Strassen block, we're multiplying a 2x2 A block by a 2x2 B block to get a 2x2 C block.
        # The A block is at rows (a_row_offset, a_row_offset+1) of the 2-row A, cols (0,1) of the 2-col A.
        # Wait, in the 2x2x10 case, A is always 2x2 (cols 0,1). So a_row_offset is 0.
        # The B block is at rows (0,1) of the 2-row B, cols (b_col_offset, b_col_offset+1).
        # The C block is at rows (0,1) of the 2-row C, cols (c_col_offset, c_col_offset+1).
        
        # U indices for A elements:
        # A[0,0] -> U[0*3+0] = U[0]
        # A[0,1] -> U[0*3+1] = U[1]
        # A[1,0] -> U[1*3+0] = U[3]
        # A[1,1] -> U[1*3+1] = U[4]
        
        # V indices for B elements:
        # B[0, b_col_offset] -> V[0*10 + b_col_offset]
        # B[0, b_col_offset+1] -> V[0*10 + b_col_offset+1]
        # B[1, b_col_offset] -> V[1*10 + b_col_offset]
        # B[1, b_col_offset+1] -> V[1*10 + b_col_offset+1]
        
        # W indices for C elements:
        # C[0, c_col_offset] -> W[c_col_offset*2 + 0]
        # C[0, c_col_offset+1] -> W[c_col_offset*2 + 1]
        # C[1, c_col_offset] -> W[c_col_offset*2 + 0 + 2]
        # C[1, c_col_offset+1] -> W[c_col_offset*2 + 1 + 2]
        
        # Strassen coefficients:
        # M1 = (A11 + A22)(B11 + B22)
        # M2 = (A21 + A22)B11
        # M3 = A11(B21 - B22)
        # M4 = A22(B12 - B11)
        # M5 = (A11 + A12)B22
        # M6 = (A21 - A11)(B11 + B12)
        # M7 = (A12 - A22)(B21 + B22)
        
        # C11 = M1 + M4 - M5 + M7
        # C12 = M3 + M5
        # C21 = M2 + M4
        # C22 = M1 - M2 + M3 + M6
        
        # Let's define the 7 multiplications and their U, V coefficients.
        
        # M1: (A11 + A22)(B11 + B22)
        # U: A11=1, A22=1 -> U[0]=1, U[4]=1
        # V: B11=1, B22=1 -> V[0*10+b_col_offset]=1, V[1*10+b_col_offset+1]=1
        # W: C11+=1, C22+=1
        col = base_col + 0
        U[0][col] += 1
        U[4][col] += 1
        V[0*10 + b_col_offset][col] += 1
        V[1*10 + b_col_offset+1][col] += 1
        W[c_col_offset*2 + 0][col] += 1
        W[c_col_offset*2 + 1 + 2][col] += 1
        
        # M2: (A21 + A22)B11
        # U: A21=1, A22=1 -> U[3]=1, U[4]=1
        # V: B11=1 -> V[0*10+b_col_offset]=1
        # W: C21+=1
        col = base_col + 1
        U[3][col] += 1
        U[4][col] += 1
        V[0*10 + b_col_offset][col] += 1
        W[c_col_offset*2 + 0 + 2][col] += 1
        
        # M3: A11(B21 - B22)
        # U: A11=1 -> U[0]=1
        # V: B21=1, B22=-1 -> V[1*10+b_col_offset]=1, V[1*10+b_col_offset+1]=-1
        # W: C12+=1, C22+=1
        col = base_col + 2
        U[0][col] += 1
        V[1*10 + b_col_offset][col] += 1
        V[1*10 + b_col_offset+1][col] += -1
        W[c_col_offset*2 + 1][col] += 1
        W[c_col_offset*2 + 1 + 2][col] += 1
        
        # M4: A22(B12 - B11)
        # U: A22=1 -> U[4]=1
        # V: B12=1, B11=-1 -> V[0*10+b_col_offset+1]=1, V[0*10+b_col_offset]=-1
        # W: C11+=1, C21+=1
        col = base_col + 3
        U[4][col] += 1
        V[0*10 + b_col_offset+1][col] += 1
        V[0*10 + b_col_offset][col] += -1
        W[c_col_offset*2 + 0][col] += 1
        W[c_col_offset*2 + 0 + 2][col] += 1
        
        # M5: (A11 + A12)B22
        # U: A11=1, A12=1 -> U[0]=1, U[1]=1
        # V: B22=1 -> V[1*10+b_col_offset+1]=1
        # W: C11-=-1, C12+=1
        col = base_col + 4
        U[0][col] += 1
        U[1][col] += 1
        V[1*10 + b_col_offset+1][col] += 1
        W[c_col_offset*2 + 0][col] += -1
        W[c_col_offset*2 + 1][col] += 1
        
        # M6: (A21 - A11)(B11 + B12)
        # U: A21=1, A11=-1 -> U[3]=1, U[0]=-1
        # V: B11=1, B12=1 -> V[0*10+b_col_offset]=1, V[0*10+b_col_offset+1]=1
        # W: C22+=1
        col = base_col + 5
        U[3][col] += 1
        U[0][col] += -1
        V[0*10 + b_col_offset][col] += 1
        V[0*10 + b_col_offset+1][col] += 1
        W[c_col_offset*2 + 1 + 2][col] += 1
        
        # M7: (A12 - A22)(B21 + B22)
        # U: A12=1, A22=-1 -> U[1]=1, U[4]=-1
        # V: B21=1, B22=1 -> V[1*10+b_col_offset]=1, V[1*10+b_col_offset+1]=1
        # W: C11+=1
        col = base_col + 6
        U[1][col] += 1
        U[4][col] += -1
        V[1*10 + b_col_offset][col] += 1
        V[1*10 + b_col_offset+1][col] += 1
        W[c_col_offset*2 + 0][col] += 1
    
    # Part 1: 2x2x10 using 5 Strassen blocks for cols (0,1), (2,3), (4,5), (6,7), (8,9)
    for block in range(5):
        b_col_offset = block * 2
        c_col_offset = block * 2
        add_strassen_block(block * 7, 0, b_col_offset, 0, c_col_offset)
    
    # Part 2: 2x1x10 using rank 20
    # A is 2x1 (col 2 of the 3-col matrix), B is 1x10 (row 2 of the 3x10 matrix)
    # C is 2x10
    # U rows for A: i*3+2 for i in {0,1} -> U[2], U[5]
    # V rows for B: 2*10+k for k in {0,...,9} -> V[20], V[21], ..., V[29]
    # W rows for C: k*2+i for k in {0,...,9}, i in {0,1}
    
    base_col = 35
    for i in range(2):
        for k in range(10):
            col = base_col + i * 10 + k
            U[i * 3 + 2][col] += 1  # A[i,2]
            V[2 * 10 + k][col] += 1  # B[2,k]
            W[k * 2 + i][col] += 1  # C[i,k]
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
