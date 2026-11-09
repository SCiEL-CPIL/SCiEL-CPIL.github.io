"""Program for matrix multiplication 6x6x6; EVOLVE-BLOCK contains the decomposition."""
DIMENSIONS = [6, 6, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # We use a block decomposition: 6 = 3 * 2
    # U has shape (n*m, rank) = (36, rank)
    # V has shape (m*p, rank) = (36, rank)
    # W has shape (p*n, rank) = (36, rank)
    
    # We combine:
    # 1. Schoolbook for 3x3x3 (rank 27) -> gives rank 27 * (2*2*2) = 27*8 = 216? 
    #    Wait, the block size for the "inner" multiplication matters.
    #    
    # Let's use a simpler approach: 
    # 6x6 = (2x3) x (3x2)? No, dimensions are n x m and m x p.
    # A is 6x6, B is 6x6.
    # We can view A as a 2x2 block matrix of 3x3 blocks? Or 3x3 block matrix of 2x2 blocks?
    # 
    # Option 1: A is 3x3 blocks of 2x2. B is 3x3 blocks of 2x2.
    # Then C is 3x3 blocks of 2x2.
    # The number of 2x2 block multiplications is 3*3*3 = 27.
    # Each 2x2 multiplication can be done with rank 8 (schoolbook) or 7 (Strassen).
    # If we use Strassen for each 2x2 block, total rank = 27 * 7 = 189.
    # 
    # Option 2: A is 2x2 blocks of 3x3. B is 2x2 blocks of 3x3.
    # Then C is 2x2 blocks of 3x3.
    # Number of 3x3 block multiplications is 2*2*2 = 8.
    # Each 3x3 multiplication has rank 27 (schoolbook). Total rank = 8 * 27 = 216.
    # 
    # Clearly Option 1 is better: 189 < 216.
    # 
    # So we use Strassen for 2x2 blocks, and there are 27 such blocks.
    # 
    # Let's define the indices.
    # A[i,j] where i in [0..5], j in [0..5].
    # Let i = 3*i0 + i1, j = 3*j0 + j1, where i0,j0 in [0..2], i1,j1 in [0..1].
    # Then A is a 3x3 block matrix A[i0,j0] where each block is 2x2.
    # Similarly B[j0,k0] where j0,k0 in [0..2], each block 2x2.
    # C[k0,i0] where k0,i0 in [0..2], each block 2x2.
    # 
    # C[i0,k0] = sum_{j0} A[i0,j0] * B[j0,k0]
    # Each product is a 2x2 matrix product, done via Strassen (7 scalar mults).
    # 
    # For each (i0,j0,k0) triple, we have a Strassen decomposition of the 2x2 product.
    # The U, V, W matrices for the whole 6x6 product will be built from these.
    # 
    # U has rows indexed by (i,j) = (3*i0+i1, 3*j0+j1).
    # V has rows indexed by (j,k) = (3*j0+j1, 3*k0+k1).
    # W has rows indexed by (k,i) = (3*k0+k1, 3*i0+i1).
    # 
    # For each (i0,j0,k0), Strassen gives 7 terms.
    # Total rank = 27 * 7 = 189.
    # 
    # Strassen for 2x2:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B01-B11)
    # M4 = A11(B10-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    # 
    # C00 = M1 - M4 + M5 - M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    # 
    # Let's define the Strassen U, V, W for 2x2.
    # U rows: (i1,j1) for i1,j1 in [0,1].
    # V rows: (j1,k1) for j1,k1 in [0,1].
    # W rows: (k1,i1) for k1,i1 in [0,1].
    
    # U_strassen: 4x7
    # V_strassen: 4x7
    # W_strassen: 4x7
    
    # M1: U: A00,A11; V: B00,B11
    # M2: U: A10,A11; V: B00
    # M3: U: A00; V: B01,B11
    # M4: U: A11; V: B10,B00
    # M5: U: A00,A01; V: B11
    # M6: U: A10,A00; V: B00,B01
    # M7: U: A01,A11; V: B10,B11
    
    U_str = [[0]*7 for _ in range(4)]
    V_str = [[0]*7 for _ in range(4)]
    W_str = [[0]*7 for _ in range(4)]
    
    # Row indices for 2x2:
    # U: (0,0)=0, (0,1)=1, (1,0)=2, (1,1)=3
    # V: (0,0)=0, (0,1)=1, (1,0)=2, (1,1)=3
    # W: (0,0)=0, (0,1)=1, (1,0)=2, (1,1)=3
    
    # M1 (col 0):
    U_str[0][0] = 1  # A00
    U_str[3][0] = 1  # A11
    V_str[0][0] = 1  # B00
    V_str[3][0] = 1  # B11
    W_str[0][0] = 1  # C00
    W_str[3][0] = 1  # C11
    
    # M2 (col 1):
    U_str[2][1] = 1  # A10
    U_str[3][1] = 1  # A11
    V_str[0][1] = 1  # B00
    W_str[2][1] = 1  # C10
    
    # M3 (col 2):
    U_str[0][2] = 1  # A00
    V_str[1][2] = 1  # B01
    V_str[3][2] = -1 # B11
    W_str[1][2] = 1  # C01
    W_str[3][2] = 1  # C11
    
    # M4 (col 3):
    U_str[3][3] = 1  # A11
    V_str[2][3] = 1  # B10
    V_str[0][3] = -1 # B00
    W_str[0][3] = -1 # C00
    W_str[2][3] = 1  # C10
    
    # M5 (col 4):
    U_str[0][4] = 1  # A00
    U_str[1][4] = 1  # A01
    V_str[3][4] = 1  # B11
    W_str[0][4] = 1  # C00
    W_str[1][4] = 1  # C01
    
    # M6 (col 5):
    U_str[2][5] = 1  # A10
    U_str[0][5] = -1 # A00
    V_str[0][5] = 1  # B00
    V_str[1][5] = 1  # B01
    W_str[3][5] = 1  # C11
    
    # M7 (col 6):
    U_str[1][6] = 1  # A01
    U_str[3][6] = -1 # A11
    V_str[2][6] = 1  # B10
    V_str[3][6] = 1  # B11
    W_str[0][6] = -1 # C00
    
    rank = 27 * 7
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    for i0 in range(3):
        for j0 in range(3):
            for k0 in range(3):
                for c in range(7):
                    # U rows: (i1, j1) -> (3*i0+i1, 3*j0+j1)
                    for i1 in range(2):
                        for j1 in range(2):
                            u_idx = (3*i0+i1) * m + (3*j0+j1)
                            U[u_idx][col] = U_str[i1*2+j1][c]
                    
                    # V rows: (j1, k1) -> (3*j0+j1, 3*k0+k1)
                    for j1 in range(2):
                        for k1 in range(2):
                            v_idx = (3*j0+j1) * p + (3*k0+k1)
                            V[v_idx][col] = V_str[j1*2+k1][c]
                    
                    # W rows: (k1, i1) -> (3*k0+k1, 3*i0+i1)
                    for k1 in range(2):
                        for i1 in range(2):
                            w_idx = (3*k0+k1) * n + (3*i0+i1)
                            W[w_idx][col] = W_str[k1*2+i1][c]
                    
                    col += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
