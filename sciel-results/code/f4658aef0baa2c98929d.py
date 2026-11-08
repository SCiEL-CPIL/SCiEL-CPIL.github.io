"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 5, 5]

# EVOLVE-BLOCK-START
import time

def _build_4x5x5_decomposition():
    """
    Construct a rank-85 exact tensor decomposition for 4x5 * 5x5 -> 4x5.
    
    Strategy:
    - Split A (4x5) into A1 (4x4) and A2 (4x1) along columns.
    - Split B (5x5) into B1 (first 4 columns, 5x4) and B2 (last column, 5x1) along columns.
      Actually, B is 5x5. We split B into B_left (5x4) and B_right (5x1).
      C = A @ B = A1 @ B_left + A2 @ B_right... no wait.
      
      A is 4x5, B is 5x5.
      A = [A1 | A2] where A1 is 4x4, A2 is 4x1.
      B = [B1; B2] where B1 is 4x5 (first 4 rows), B2 is 1x5 (last row).
      C = A1 @ B1 + A2 @ B2
      
      A1 @ B1: 4x4 * 4x5 -> 4x5
      A2 @ B2: 4x1 * 1x5 -> 4x5
      
      For A1 @ B1 (4x4 * 4x5):
      Split B1 into B1a (4x4) and B1b (4x1).
      A1 @ B1 = A1 @ B1a + A1 @ B1b
      
      A1 @ B1a: 4x4 * 4x4 -> 4x4. Use Strassen-like rank 49? 
      Actually, let's use the known rank 49 for 4x4x4.
      But implementing rank 49 from scratch is complex. 
      Let's try a different splitting to use simpler blocks.
      
      Alternative: Use rank 100 schoolbook as fallback, but try to get a better one.
      The task says "rank-minimal". 85 is better than 100.
      
      Let's implement the rank 85 construction:
      1. 4x4 * 4x4 part: Use Strassen decomposition recursively.
         Strassen for 2x2x2 has rank 7.
         4x4x4 can be done with 7 multiplications of 2x2x2, each of rank 7, so 7*7=49.
      2. 4x4 * 4x1 part: Schoolbook, rank 16.
      3. 4x1 * 1x5 part: Schoolbook, rank 20.
      Total: 49 + 16 + 20 = 85.
      
      We need to implement the 4x4x4 rank 49 decomposition.
      The 2x2x2 Strassen decomposition:
      U, V, W matrices of size 4x7, 4x7, 4x7 for n=m=p=2.
      
      For 4x4x4, we treat the matrices as 2x2 blocks of 2x2 submatrices.
      The decomposition is the "tensor product" of the 2x2x2 Strassen with itself? 
      Not exactly. It's applying Strassen at the block level, where each block multiplication is done via Strassen.
      
      Let's define the 2x2x2 Strassen factors.
      n=m=p=2.
      Indices for U: (i,j) with i in 0..1, j in 0..1. Row index i*2+j.
      Indices for V: (j,k) with j in 0..1, k in 0..1. Row index j*2+k.
      Indices for W: (k,i) with k in 0..1, i in 0..1. Row index k*2+i.
      
      Strassen factors (one of many valid ones):
      M1 = (A11+A22)(B11+B22)
      M2 = (A21+A22)B11
      M3 = A11(B12-B22)
      M4 = A22(B21-B11)
      M5 = (A11+A12)B22
      M6 = (A21-A11)(B11+B12)
      M7 = (A12-A22)(B21+B22)
      
      C11 = M1 - M2 + M4 + M6
      C12 = M3 + M5
      C21 = M2 + M7
      C22 = M1 - M3 + M5 - M7
      
      U (4x7):
      col 0 (M1): A11=1, A22=1 -> U[0,0]=1, U[3,0]=1
      col 1 (M2): A21=1, A22=1 -> U[2,1]=1, U[3,1]=1
      col 2 (M3): A11=1 -> U[0,2]=1
      col 3 (M4): A22=1 -> U[3,3]=1
      col 4 (M5): A11=1, A12=1 -> U[0,4]=1, U[1,4]=1
      col 5 (M6): A21=1, A11=-1 -> U[2,5]=1, U[0,5]=-1
      col 6 (M7): A12=1, A22=-1 -> U[1,6]=1, U[3,6]=-1
      
      V (4x7):
      col 0 (M1): B11=1, B22=1 -> V[0,0]=1, V[3,0]=1
      col 1 (M2): B11=1 -> V[0,1]=1
      col 2 (M3): B12=1, B22=-1 -> V[1,2]=1, V[3,2]=-1
      col 3 (M4): B21=1, B11=-1 -> V[2,3]=1, V[0,3]=-1
      col 4 (M5): B22=1 -> V[3,4]=1
      col 5 (M6): B11=1, B12=1 -> V[0,5]=1, V[1,5]=1
      col 6 (M7): B21=1, B22=1 -> V[2,6]=1, V[3,6]=1
      
      W (4x7):
      C11 = M1 - M2 + M4 + M6 -> W[0,0]=1, W[0,1]=-1, W[0,3]=1, W[0,5]=1
      C12 = M3 + M5 -> W[1,2]=1, W[1,4]=1
      C21 = M2 + M7 -> W[2,1]=1, W[2,6]=1
      C22 = M1 - M3 + M5 - M7 -> W[3,0]=1, W[3,2]=-1, W[3,4]=1, W[3,6]=-1
      
      Now for 4x4x4, we use this as a template.
      The 4x4x4 problem has n=m=p=4.
      We view A as 2x2 blocks of 2x2.
      A = [[A11, A12], [A21, A22]] where each Aij is 2x2.
      Similarly for B and C.
      
      The Strassen formula operates on these blocks.
      M1_block = (A11_block + A22_block) * (B11_block + B22_block)
      This block multiplication is a 2x2 * 2x2, which we do with the rank 7 Strassen.
      So each "M" in the 4x4x4 Strassen is itself a 2x2x2 Strassen with rank 7.
      Total rank = 7 * 7 = 49.
      
      We need to construct U, V, W for 4x4x4.
      U is 16x49. V is 16x49. W is 16x49.
      
      For each of the 7 outer Strassen columns (col_s for s in 0..6),
      we have a 2x2x2 Strassen inside.
      For each inner column (col_t for t in 0..6),
      the global column index is col_s * 7 + col_t.
      
      U[global_row, global_col] = U_outer[i_block, j_block, col_s] * U_inner[i_sub, j_sub, col_t]
      where global_row = i_block * 4 + j_block * 2 + i_sub? No.
      A is 4x4. Block (i,j) where i,j in 0..1. Each block is 2x2.
      Row index in A: i_block * 2 + i_sub, where i_block in 0..1, i_sub in 0..1.
      Col index in A: j_block * 2 + j_sub.
      So U row index = (i_block * 2 + i_sub) * 4 + (j_block * 2 + j_sub).
      
      U_outer[i_block, j_block, col_s] is the Strassen U factor for the block level.
      U_inner[i_sub, j_sub, col_t] is the Strassen U factor for the sub-block level.
      
      Similarly for V and W.
      
      Let's code this up.
    """
    # 2x2x2 Strassen factors
    U2 = [[0]*7 for _ in range(4)]
    V2 = [[0]*7 for _ in range(4)]
    W2 = [[0]*7 for _ in range(4)]
    
    # U2
    U2[0][0] = 1; U2[3][0] = 1
    U2[2][1] = 1; U2[3][1] = 1
    U2[0][2] = 1
    U2[3][3] = 1
    U2[0][4] = 1; U2[1][4] = 1
    U2[2][5] = 1; U2[0][5] = -1
    U2[1][6] = 1; U2[3][6] = -1
    
    # V2
    V2[0][0] = 1; V2[3][0] = 1
    V2[0][1] = 1
    V2[1][2] = 1; V2[3][2] = -1
    V2[2][3] = 1; V2[0][3] = -1
    V2[3][4] = 1
    V2[0][5] = 1; V2[1][5] = 1
    V2[2][6] = 1; V2[3][6] = 1
    
    # W2
    W2[0][0] = 1; W2[0][1] = -1; W2[0][3] = 1; W2[0][5] = 1
    W2[1][2] = 1; W2[1][4] = 1
    W2[2][1] = 1; W2[2][6] = 1
    W2[3][0] = 1; W2[3][2] = -1; W2[3][4] = 1; W2[3][6] = -1
    
    # Build 4x4x4 rank 49
    U4 = [[0]*49 for _ in range(16)]
    V4 = [[0]*49 for _ in range(16)]
    W4 = [[0]*49 for _ in range(16)]
    
    for s in range(7):  # outer column
        for t in range(7):  # inner column
            gc = s * 7 + t
            # U
            for ib in range(2):
                for jb in range(2):
                    u_out = U2[ib*2+jb][s]
                    if u_out != 0:
                        for isub in range(2):
                            for jsub in range(2):
                                u_in = U2[isub*2+jsub][t]
                                if u_in != 0:
                                    row = (ib*2+isub)*4 + (jb*2+jsub)
                                    U4[row][gc] += u_out * u_in
            # V
            for jb in range(2):
                for kb in range(2):
                    v_out = V2[jb*2+kb][s]
                    if v_out != 0:
                        for jsub in range(2):
                            for ksub in range(2):
                                v_in = V2[jsub*2+ksub][t]
                                if v_in != 0:
                                    row = (jb*2+jsub)*4 + (kb*2+ksub)
                                    V4[row][gc] += v_out * v_in
            # W
            for kb in range(2):
                for ib in range(2):
                    w_out = W2[kb*2+ib][s]
                    if w_out != 0:
                        for ksub in range(2):
                            for isub in range(2):
                                w_in = W2[ksub*2+isub][t]
                                if w_in != 0:
                                    row = (kb*2+ksub)*4 + (ib*2+isub)
                                    W4[row][gc] += w_out * w_in
    
    # Now combine for 4x5x5
    # n=4, m=5, p=5
    # U: 4x5 = 20 rows. V: 5x5 = 25 rows. W: 5x4 = 20 rows.
    # Total rank = 49 + 16 + 20 = 85
    
    rank = 85
    U = [[0]*rank for _ in range(20)]
    V = [[0]*rank for _ in range(25)]
    W = [[0]*rank for _ in range(20)]
    
    col_offset = 0
    
    # Part 1: A1 (4x4) * B1a (4x4) -> C1 (4x4)
    # A1 columns 0-3, B1a columns 0-3
    # A1 rows 0-3, B1a rows 0-3
    # This uses the 4x4x4 decomposition U4, V4, W4
    for r in range(16):
        for c in range(49):
            if U4[r][c] != 0:
                U[r][col_offset + c] = U4[r][c]
            if V4[r][c] != 0:
                V[r][col_offset + c] = V4[r][c]
            if W4[r][c] != 0:
                W[r][col_offset + c] = W4[r][c]
    col_offset += 49
    
    # Part 2: A1 (4x4) * B1b (4x1) -> C2 (4x1)
    # B1b is column 4 of B1, which is column 4 of B.
    # Schoolbook: rank 16
    # A1 is rows 0-3, cols 0-3 of A.
    # B1b is rows 0-3, col 4 of B.
    # C2 is rows 0-3, col 4 of C.
    for i in range(4):
        for j in range(4):
            U[i*5 + j][col_offset] = 1
            V[j*5 + 4][col_offset] = 1
            W[4*4 + i][col_offset] = 1  # W index: k*n + i, k=4 (col index), i=row
            col_offset += 1
    
    # Part 3: A2 (4x1) * B2 (1x5) -> C3 (4x5)
    # A2 is col 4 of A. B2 is row 4 of B.
    # Schoolbook: rank 20
    # A2: rows 0-3, col 4.
    # B2: row 4, cols 0-4.
    # C3: rows 0-3, cols 0-4.
    for i in range(4):
        for k in range(5):
            U[i*5 + 4][col_offset] = 1
            V[4*5 + k][col_offset] = 1
            W[k*4 + i][col_offset] = 1
            col_offset += 1
    
    return {"U": U, "V": V, "W": W}

def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    
    # Try to build the rank-85 decomposition
    try:
        result = _build_4x5x5_decomposition()
        elapsed = time.time() - start_time
        if elapsed < deadline:
            return result
    except Exception as e:
        pass
    
    # Fallback to schoolbook rank 100
    n, m, p = DIMENSIONS
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
