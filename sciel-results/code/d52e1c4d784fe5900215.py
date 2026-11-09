"""Matrix multiplication 2x5x6 rank-minimal exact tensor decomposition."""
import time
import numpy as np

DIMENSIONS = [2, 5, 6]

# EVOLVE-BLOCK-START

def _strassen_2x2x2():
    """Return U, V, W for 2x2x2 Strassen decomposition.
    U: (4, 7), V: (4, 7), W: (4, 7)
    Indices: U[i*2+j, c] for A[i,j], V[i*2+j, c] for B[i,j], W[i*2+j, c] for C[i,j]
    """
    # Strassen's 7 multiplications
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B01-B11)
    # M4 = A11(B10-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B10)
    # M7 = (A01-A11)(B01+B11)
    
    # U rows: A00, A01, A10, A11 (indices 0,1,2,3)
    # V rows: B00, B01, B10, B11 (indices 0,1,2,3)
    # W rows: C00, C01, C10, C11 (indices 0,1,2,3)
    
    U = np.zeros((4, 7))
    V = np.zeros((4, 7))
    W = np.zeros((4, 7))
    
    # M1: (A00+A11)(B00+B11)
    U[0, 0] = 1  # A00
    U[3, 0] = 1  # A11
    V[0, 0] = 1  # B00
    V[3, 0] = 1  # B11
    
    # M2: (A10+A11)B00
    U[2, 1] = 1  # A10
    U[3, 1] = 1  # A11
    V[0, 1] = 1  # B00
    
    # M3: A00(B01-B11)
    U[0, 2] = 1  # A00
    V[1, 2] = 1  # B01
    V[3, 2] = -1 # B11
    
    # M4: A11(B10-B00)
    U[3, 3] = 1  # A11
    V[2, 3] = 1  # B10
    V[0, 3] = -1 # B00
    
    # M5: (A00+A01)B11
    U[0, 4] = 1  # A00
    U[1, 4] = 1  # A01
    V[3, 4] = 1  # B11
    
    # M6: (A10-A00)(B00+B10)
    U[2, 5] = 1  # A10
    U[0, 5] = -1 # A00
    V[0, 5] = 1  # B00
    V[2, 5] = 1  # B10
    
    # M7: (A01-A11)(B01+B11)
    U[1, 6] = 1  # A01
    U[3, 6] = -1 # A11
    V[1, 6] = 1  # B01
    V[3, 6] = 1  # B11
    
    # W: reconstruct C from M1..M7
    # C00 = M1+M4-M5+M7
    W[0, 0] = 1
    W[0, 3] = 1
    W[0, 4] = -1
    W[0, 6] = 1
    
    # C01 = M3+M5
    W[1, 2] = 1
    W[1, 4] = 1
    
    # C10 = M2+M4
    W[2, 1] = 1
    W[2, 3] = 1
    
    # C11 = M1-M2+M3+M6
    W[3, 0] = 1
    W[3, 1] = -1
    W[3, 2] = 1
    W[3, 5] = 1
    
    return U, V, W

def _schoolbook_block(n, m, p, offset=0):
    """Return U, V, W for schoolbook n x m x p block.
    U: (n*m, n*m*p), V: (m*p, n*m*p), W: (p*n, n*m*p)
    """
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    return U, V, W

def _verify(U, V, W, n, m, p, tol=1e-8):
    """Verify the decomposition."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    rank = U.shape[1]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = np.sum(U[i*m+j, :] * V[j*p+k, :] * W[k*n+i, :])
                if abs(s - 1.0) > tol:
                    return False
    return True

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140.0
    
    # Strategy: Split inner dimension m=5 into 2+3
    # A: 2x5 = [A1 (2x2) | A2 (2x3)]
    # B: 5x6 = [B1 (2x6); B2 (3x6)]
    # C = A1@B1 + A2@B2
    #
    # A1@B1: 2x2x6. Split p=6 into 2+2+2
    #   3 sub-problems of 2x2x2, each using Strassen (rank 7)
    # A2@B2: 2x3x6. Split p=6 into 2+2+2  
    #   3 sub-problems of 2x3x2, each using schoolbook (rank 12)
    #
    # Total rank: 3*7 + 3*12 = 21 + 36 = 57
    
    # Build the decomposition
    # U rows: A has 2*5=10 rows. 
    #   A1 rows: indices 0,1 (A[0,0], A[0,1]) -> wait, A is 2x5
    #   A[i,j] for i in 0..1, j in 0..4
    #   A1: A[0,0], A[0,1], A[1,0], A[1,1] -> rows 0,1,2,3 of U
    #   A2: A[0,2], A[0,3], A[0,4], A[1,2], A[1,3], A[1,4] -> rows 4,5,6,7,8,9 of U
    #
    # V rows: B has 5*6=30 rows.
    #   B[j,k] for j in 0..4, k in 0..5
    #   B1: B[0,0..5], B[1,0..5] -> rows 0..11 of V (12 rows)
    #   B2: B[2,0..5], B[3,0..5], B[4,0..5] -> rows 12..29 of V (18 rows)
    #
    # W rows: C has 6*2=12 rows (k*n+i for k in 0..5, i in 0..1)
    #   C[i,k] -> W[k*2+i, c]
    
    # Let's build it piece by piece
    # Part 1: A1@B1 where A1 is 2x2, B1 is 2x6
    # Split B1 into 3 blocks of 2x2: B1_1 (cols 0-1), B1_2 (cols 2-3), B1_3 (cols 4-5)
    # Each 2x2x2 uses Strassen rank 7
    
    # Part 2: A2@B2 where A2 is 2x3, B2 is 3x6
    # Split B2 into 3 blocks of 3x2: B2_1 (cols 0-1), B2_2 (cols 2-3), B2_3 (cols 4-5)
    # Each 2x3x2 uses schoolbook rank 12
    
    total_rank = 3 * 7 + 3 * 12  # 21 + 36 = 57
    
    U_full = np.zeros((10, total_rank))
    V_full = np.zeros((30, total_rank))
    W_full = np.zeros((12, total_rank))
    
    col_offset = 0
    
    # Part 1: A1@B1, 3 Strassen blocks
    U_str, V_str, W_str = _strassen_2x2x2()
    
    for block in range(3):
        # Block b handles B1 columns 2*b to 2*b+1, and C columns 2*b to 2*b+1
        # A1 rows: A[0,0], A[0,1], A[1,0], A[1,1] -> U rows 0,1,2,3
        # B1 rows for this block: B[0,2*b], B[0,2*b+1], B[1,2*b], B[1,2*b+1]
        #   B[j,k] -> V row j*6+k
        #   B[0,2*b] -> V row 0*6+2*b = 2*b
        #   B[0,2*b+1] -> V row 0*6+(2*b+1) = 2*b+1
        #   B[1,2*b] -> V row 1*6+2*b = 6+2*b
        #   B[1,2*b+1] -> V row 1*6+(2*b+1) = 6+2*b+1
        # C rows for this block: C[0,2*b], C[0,2*b+1], C[1,2*b], C[1,2*b+1]
        #   C[i,k] -> W row k*2+i
        #   C[0,2*b] -> W row (2*b)*2+0 = 4*b
        #   C[0,2*b+1] -> W row (2*b+1)*2+0 = 4*b+2
        #   C[1,2*b] -> W row (2*b)*2+1 = 4*b+1
        #   C[1,2*b+1] -> W row (2*b+1)*2+1 = 4*b+3
        
        # Copy Strassen U to U_full rows 0-3
        U_full[0:4, col_offset:col_offset+7] = U_str
        
        # Copy Strassen V to appropriate V_full rows
        # V_str row 0 -> B[0,0] -> V_full row 2*b
        # V_str row 1 -> B[0,1] -> V_full row 2*b+1
        # V_str row 2 -> B[1,0] -> V_full row 6+2*b
        # V_str row 3 -> B[1,1] -> V_full row 6+2*b+1
        V_full[2*b, col_offset:col_offset+7] = V_str[0, :]
        V_full[2*b+1, col_offset:col_offset+7] = V_str[1, :]
        V_full[6+2*b, col_offset:col_offset+7] = V_str[2, :]
        V_full[6+2*b+1, col_offset:col_offset+7] = V_str[3, :]
        
        # Copy Strassen W to appropriate W_full rows
        # W_str row 0 -> C[0,0] -> W_full row 4*b
        # W_str row 1 -> C[0,1] -> W_full row 4*b+2
        # W_str row 2 -> C[1,0] -> W_full row 4*b+1
        # W_str row 3 -> C[1,1] -> W_full row 4*b+3
        W_full[4*b, col_offset:col_offset+7] = W_str[0, :]
        W_full[4*b+2, col_offset:col_offset+7] = W_str[1, :]
        W_full[4*b+1, col_offset:col_offset+7] = W_str[2, :]
        W_full[4*b+3, col_offset:col_offset+7] = W_str[3, :]
        
        col_offset += 7
    
    # Part 2: A2@B2, 3 schoolbook blocks of 2x3x2
    # A2 rows: A[0,2], A[0,3], A[0,4], A[1,2], A[1,3], A[1,4] -> U rows 4,5,6,7,8,9
    # B2: B[2,0..5], B[3,0..5], B[4,0..5] -> V rows 12..29
    # For block b (cols 2*b to 2*b+1):
    #   B2 block: B[2,2*b], B[2,2*b+1], B[3,2*b], B[3,2*b+1], B[4,2*b], B[4,2*b+1]
    #   V rows: 12+2*b, 12+2*b+1, 18+2*b, 18+2*b+1, 24+2*b, 24+2*b+1
    #   C rows: same as before, 4*b to 4*b+3
    
    for block in range(3):
        # Schoolbook for 2x3x2: rank = 2*3*2 = 12
        # U_sub: 6 rows (A2), V_sub: 6 rows (B2 block), W_sub: 4 rows (C block)
        U_sub, V_sub, W_sub = _schoolbook_block(2, 3, 2)
        
        # A2 rows in U_full: 4,5,6,7,8,9
        # A2[i,j] where i=0,1 and j=2,3,4
        # U_sub row 0 -> A2[0,0] -> A[0,2] -> U_full row 4
        # U_sub row 1 -> A2[0,1] -> A[0,3] -> U_full row 5
        # U_sub row 2 -> A2[0,2] -> A[0,4] -> U_full row 6
        # U_sub row 3 -> A2[1,0] -> A[1,2] -> U_full row 7
        # U_sub row 4 -> A2[1,1] -> A[1,3] -> U_full row 8
        # U_sub row 5 -> A2[1,2] -> A[1,4] -> U_full row 9
        U_full[4:10, col_offset:col_offset+12] = U_sub
        
        # B2 block rows in V_full:
        # B2[i,j] where i=2,3,4 and j=2*b, 2*b+1
        # V_sub row 0 -> B2[0,0] -> B[2,2*b] -> V_full row 2*6+2*b = 12+2*b
        # V_sub row 1 -> B2[0,1] -> B[2,2*b+1] -> V_full row 12+2*b+1
        # V_sub row 2 -> B2[1,0] -> B[3,2*b] -> V_full row 3*6+2*b = 18+2*b
        # V_sub row 3 -> B2[1,1] -> B[3,2*b+1] -> V_full row 18+2*b+1
        # V_sub row 4 -> B2[2,0] -> B[4,2*b] -> V_full row 4*6+2*b = 24+2*b
        # V_sub row 5 -> B2[2,1] -> B[4,2*b+1] -> V_full row 24+2*b+1
        V_full[12+2*b, col_offset:col_offset+12] = V_sub[0, :]
        V_full[12+2*b+1, col_offset:col_offset+12] = V_sub[1, :]
        V_full[18+2*b, col_offset:col_offset+12] = V_sub[2, :]
        V_full[18+2*b+1, col_offset:col_offset+12] = V_sub[3, :]
        V_full[24+2*b, col_offset:col_offset+12] = V_sub[4, :]
        V_full[24+2*b+1, col_offset:col_offset+12] = V_sub[5, :]
        
        # C rows: same as Part 1
        W_full[4*b, col_offset:col_offset+12] = W_sub[0, :]
        W_full[4*b+2, col_offset:col_offset+12] = W_sub[1, :]
        W_full[4*b+1, col_offset:col_offset+12] = W_sub[2, :]
        W_full[4*b+3, col_offset:col_offset+12] = W_sub[3, :]
        
        col_offset += 12
    
    # Verify
    U_list = U_full.tolist()
    V_list = V_full.tolist()
    W_list = W_full.tolist()
    
    if _verify(U_list, V_list, W_list, n, m, p):
        return {"U": U_list, "V": V_list, "W": W_list}
    
    # Fallback: schoolbook
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
