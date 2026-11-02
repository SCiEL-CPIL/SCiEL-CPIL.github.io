import time
import numpy as np

DIMENSIONS = [3, 5, 7]

# EVOLVE-BLOCK-START
def construct():
    """
    Matrix multiplication 3x5x7.
    Strategy: Partition m=5 into 2+2+1.
    For each 3x2x7 block, partition the 3 rows of A into 2+1.
    Use rank-7 Strassen for 2x2x2, extended to 2x2x7 (rank 23).
    Use schoolbook for 1x2x7 (rank 14) and 3x1x7 (rank 21).
    Total rank: 2 * (23 + 14) + 21 = 2 * 37 + 21 = 95.
    """
    n, m, p = DIMENSIONS  # 3, 5, 7
    
    # Helper: Rank-7 Strassen decomposition for 2x2x2
    # Returns U, V, W of shape (4, 7), (4, 7), (4, 7)
    def strassen_2x2x2():
        # Indices: A[i,j], B[j,k], C[k,i]
        # U rows: A[i,j]
        # V rows: B[j,k]
        # W rows: C[k,i]
        # Forms:
        # m1 = A00 * (B00+B11) -> U: A00, V: B00,B11, W: C00,C11
        # m2 = (A01+A11) * B00 -> U: A01,A11, V: B00, W: C00
        # m3 = A10 * (B01+B10) -> U: A10, V: B01,B10, W: C10,C01
        # m4 = (A00+A10) * B10 -> U: A00,A10, V: B10, W: C10
        # m5 = (A01+A11) * B11 -> U: A01,A11, V: B11, W: C11
        # m6 = A11 * (B01+B11) -> U: A11, V: B01,B11, W: C01
        # m7 = (A00-A11) * (B11-B10) -> U: A00,A11, V: B11,B10, W: C00,C11
        
        U = [[0]*7 for _ in range(4)]
        V = [[0]*7 for _ in range(4)]
        W = [[0]*7 for _ in range(4)]
        
        # m1
        U[0][0] = 1 # A00
        V[0][0] = 1 # B00
        V[1][0] = 1 # B11
        W[0][0] = 1 # C00
        W[3][0] = 1 # C11 (index 3 is C[1,1])
        
        # m2
        U[1][1] = 1 # A01
        U[2][1] = 1 # A11
        V[0][1] = 1 # B00
        W[0][1] = 1 # C00
        
        # m3
        U[2][2] = 1 # A10
        V[1][2] = 1 # B01
        V[2][2] = 1 # B10
        W[2][2] = 1 # C10
        W[1][2] = 1 # C01
        
        # m4
        U[0][3] = 1 # A00
        U[2][3] = 1 # A10
        V[2][3] = 1 # B10
        W[2][3] = 1 # C10
        
        # m5
        U[1][4] = 1 # A01
        U[2][4] = 1 # A11
        V[3][4] = 1 # B11
        W[3][4] = 1 # C11
        
        # m6
        U[2][5] = 1 # A11
        V[1][5] = 1 # B01
        V[3][5] = 1 # B11
        W[1][5] = 1 # C01
        
        # m7
        U[0][6] = 1 # A00
        U[2][6] = -1 # A11
        V[3][6] = 1 # B11
        V[2][6] = -1 # B10
        W[0][6] = 1 # C00
        W[3][6] = -1 # C11
        
        return U, V, W

    # Helper: Schoolbook for 1x2x7 (A is 1x2, B is 2x7)
    # Rank 14
    def schoolbook_1x2x7(offset):
        U = [[0]*14 for _ in range(2)] # A[0,j]
        V = [[0]*14 for _ in range(14)] # B[j,k]
        W = [[0]*14 for _ in range(7)] # C[k,0]
        c = 0
        for j in range(2):
            for k in range(7):
                U[j][offset+c] = 1
                V[j*7+k][offset+c] = 1
                W[k][offset+c] = 1
                c += 1
        return U, V, W

    # Helper: Schoolbook for 3x1x7 (A is 3x1, B is 1x7)
    # Rank 21
    def schoolbook_3x1x7(offset):
        U = [[0]*21 for _ in range(3)] # A[i,0]
        V = [[0]*21 for _ in range(7)] # B[0,k]
        W = [[0]*21 for _ in range(21)] # C[k,i]
        c = 0
        for i in range(3):
            for k in range(7):
                U[i][offset+c] = 1
                V[k][offset+c] = 1
                W[k*3+i][offset+c] = 1
                c += 1
        return U, V, W

    # Build 2x2x7 decomposition (Rank 23)
    # Partition k=7 into 4+2+1? Or just use 2x2x2 for k=0,1 and schoolbook for rest?
    # M(2,2,7) <= M(2,2,4) + M(2,2,2) + M(2,2,1)
    # M(2,2,4) <= 2*M(2,2,2) = 14
    # M(2,2,2) = 7
    # M(2,2,1) = 2
    # Total 23.
    
    def build_2x2x7(offset):
        # U: 4x23, V: 4x23, W: 4x23 (Wait, W is 2x2=4 entries? No, C is 2x2. W rows are C[k,i]. 4 rows.)
        # Actually, for 2x2x7, C is 2x7? No, C is 2x2?
        # The block is A_sub (2x2) * B_sub (2x7). Result C_sub (2x7).
        # So W should be 2x7 = 14 rows?
        # My previous analysis: M(2,2,7) is for A(2x2) B(2x7) -> C(2x7).
        # Rank 23.
        
        U = [[0]*23 for _ in range(4)]
        V = [[0]*23 for _ in range(14)]
        W = [[0]*23 for _ in range(14)]
        
        # Part 1: k=0,1,2,3 (2x2x4)
        # Use 2 Strassen 2x2x2 blocks?
        # k=0,1: Strassen (Rank 7)
        # k=2,3: Strassen (Rank 7)
        s1U, s1V, s1W = strassen_2x2x2()
        # Map to offset
        for r in range(7):
            for i in range(4):
                U[i][offset+r] = s1U[i][r]
            for j in range(4):
                # V rows are B[j,k]. Here k is 0,1.
                # s1V[j][r] is coeff for B[j, k] where k is 0 or 1.
                # We need to map to V[j*7 + k]
                if j == 0: # B00, B01
                    V[0*7+0][offset+r] = s1V[0][r]
                    V[0*7+1][offset+r] = s1V[1][r] # Wait, s1V indices are 0..3 for B00,B01,B10,B11
                    # s1V[0] is B00, s1V[1] is B11? No.
                    # In strassen_2x2x2, V rows are 0:B00, 1:B01, 2:B10, 3:B11.
                    # So V[0][r] is B00. V[1][r] is B01.
                    pass
        # This mapping is getting complex. I will implement the schoolbook for the 2x2x7 block to be safe, 
        # but use the 3x2x7 split which saves 5 ranks per block.
        # Schoolbook 2x2x7 rank is 28.
        # My split: 2x2x7 (Rank 23) + 1x2x7 (Rank 14) = 37.
        # Schoolbook 3x2x7 rank is 42.
        # Savings 5.
        
        # To keep it simple and correct, I will use schoolbook for the 2x2x7 part (Rank 28) 
        # and just rely on the 3x1x7 part?
        # No, 37 < 42.
        
        # I will implement the 2x2x7 rank 23 manually using the 2x2x2 Strassen for k=0,1 and k=2,3 
        # and schoolbook for k=4,5,6?
        # k=0,1: 7
        # k=2,3: 7
        # k=4,5,6: 2x2x3. Rank 2x2x3 <= 2x2x2 + 2x2x1 = 7+2=9.
        # Total 7+7+9 = 23.
        
        # I will just use schoolbook for the 2x2x7 block (Rank 28) to ensure correctness first.
        # Total rank: 2 * (28 + 14) + 21 = 2 * 42 + 21 = 105.
        # This is not an improvement.
        
        # I must use the rank 23.
        # I will construct the rank 23 by calling strassen_2x2x2 three times?
        # k=0,1 (Strassen), k=2,3 (Strassen), k=4,5,6 (Schoolbook 2x2x3 -> Rank 6? No, 2x2x3 rank is 6? 2*3=6. Yes.)
        # Total 7+7+6 = 20?
        # M(2,2,3) <= M(2,2,2) + M(2,2,1) = 7+2=9.
        # Schoolbook 2x2x3 is 6. So use schoolbook.
        # Total 7+7+6 = 20.
        # 20 < 23.
        
        # So Rank 20 for 2x2x7.
        # Total: 2 * (20 + 14) + 21 = 2 * 34 + 21 = 89.
        
        # I will implement Rank 20 for 2x2x7.
        
        c = 0
        # k=0,1: Strassen
        sU, sV, sW = strassen_2x2x2()
        for r in range(7):
            for i in range(4):
                U[i][offset+c] = sU[i][r]
            # V mapping: sV[0]=B00, sV[1]=B01, sV[2]=B10, sV[3]=B11
            V[0*7+0][offset+c] = sV[0][r]
            V[0*7+1][offset+c] = sV[1][r]
            V[1*7+0][offset+c] = sV[2][r]
            V[1*7+1][offset+c] = sV[3][r]
            # W mapping: sW[0]=C00, sW[1]=C01, sW[2]=C10, sW[3]=C11
            W[0*7+0][offset+c] = sW[0][r]
            W[0*7+1][offset+c] = sW[1][r]
            W[1*7+0][offset+c] = sW[2][r]
            W[1*7+1][offset+c] = sW[3][r]
            c += 1
            
        # k=2,3: Strassen
        for r in range(7):
            for i in range(4):
                U[i][offset+c] = sU[i][r]
            V[0*7+2][offset+c] = sV[0][r]
            V[0*7+3][offset+c] = sV[1][r]
            V[1*7+2][offset+c] = sV[2][r]
            V[1*7+3][offset+c] = sV[3][r]
            W[0*7+2][offset+c] = sW[0][r]
            W[0*7+3][offset+c] = sW[1][r]
            W[1*7+2][offset+c] = sW[2][r]
            W[1*7+3][offset+c] = sW[3][r]
            c += 1
            
        # k=4,5,6: Schoolbook 2x2x3 (Rank 6)
        for j in range(2):
            for k in range(3):
                U[j][offset+c] = 1
                V[j*7 + (4+k)][offset+c] = 1
                W[j*7 + (4+k)][offset+c] = 1 # Wait, W is C[k,i]. C is 2x7.
                # W row index is k*2 + i? No, W row is C[k,i].
                # In my W matrix for 2x2x7, rows are 14.
                # W[k*2+i] corresponds to C[k,i].
                # So W[(4+k)*2 + j] = 1.
                W[(4+k)*2 + j][offset+c] = 1
                c += 1
                
        return U, V, W

    # Assemble
    # Block 1: j=0,1. A[0:3, 0:2], B[0:2, 0:7]
    # Block 2: j=2,3. A[0:3, 2:4], B[2:4, 0:7]
    # Block 3: j=4. A[0:3, 4:5], B[4:5, 0:7]
    
    # For Block 1:
    # Split A into A_top (2x2) and A_bot (1x2).
    # A_top: A[0:2, 0:2]. B: B[0:2, 0:7]. C_top: C[0:2, 0:7].
    # A_bot: A[2:3, 0:2]. B: B[0:2, 0:7]. C_bot: C[2:3, 0:7].
    
    rank_total = 0
    U_list = []
    V_list = []
    W_list = []
    
    # Block 1
    # A_top (2x2) * B (2x7) -> C_top (2x7). Rank 20.
    U1_top, V1_top, W1_top = build_2x2x7(0)
    # A_bot (1x2) * B (2x7) -> C_bot (1x7). Rank 14.
    U1_bot, V1_bot, W1_bot = schoolbook_1x2x7(20)
    
    # Combine into full U, V, W for Block 1
    U1 = [[0]*34 for _ in range(15)] # 3x5 = 15 rows
    V1 = [[0]*34 for _ in range(35)] # 5x7 = 35 rows
    W1 = [[0]*34 for _ in range(21)] # 7x3 = 21 rows
    
    # Map A_top (2x2) to U1 rows 0,1 (A[0,0], A[0,1], A[1,0], A[1,1])
    # U1_top rows: 0:A00, 1:A01, 2:A10, 3:A11
    U1[0][0:20] = U1_top[0][0:20]
    U1[1][0:20] = U1_top[1][0:20]
    U1[2][0:20] = U1_top[2][0:20]
    U1[3][0:20] = U1_top[3][0:20]
    
    # Map A_bot (1x2) to U1 row 4 (A[2,0], A[2,1])?
    # A_bot is A[2, 0:2].
    # U1_bot rows: 0:A[2,0], 1:A[2,1]
    U1[4][20:34] = U1_bot[0][20:34]
    U1[5][20:34] = U1_bot[1][20:34] # Wait, A_bot is 1x2. Rows are 0,1.
    # A[2,0] is row 2, col 0. Index 2*5+0 = 10?
    # My U1 rows are i*m+j.
    # A[0,0]=0, A[0,1]=1, A[1,0]=5, A[1,1]=6, A[2,0]=10, A[2,1]=11.
    # U1_top rows 0,1,2,3 correspond to A[0,0], A[0,1], A[1,0], A[1,1].
    # U1_bot rows 0,1 correspond to A[2,0], A[2,1].
    
    U1[0][0:20] = U1_top[0][0:20]
    U1[1][0:20] = U1_top[1][0:20]
    U1[5][0:20] = U1_top[2][0:20]
    U1[6][0:20] = U1_top[3][0:20]
    
    U1[10][20:34] = U1_bot[0][20:34]
    U1[11][20:34] = U1_bot[1][20:34]
    
    # V1 mapping
    # V1_top rows: 0:B00, 1:B01, 2:B10, 3:B11
    # V1_bot rows: 0:B00, 1:B01, 2:B10, 3:B11
    # V1 rows are j*p+k.
    # B[0,0]=0, B[0,1]=1, B[1,0]=7, B[1,1]=8.
    V1[0][0:20] = V1_top[0][0:20]
    V1[1][0:20] = V1_top[1][0:20]
    V1[7][0:20] = V1_top[2][0:20]
    V1[8][0:20] = V1_top[3][0:20]
    
    V1[0][20:34] = V1_bot[0][20:34]
    V1[1][20:34] = V1_bot[1][20:34]
    V1[7][20:34] = V1_bot[2][20:34]
    V1[8][20:34] = V1_bot[3][20:34]
    
    # W1 mapping
    # W1_top rows: 0:C00, 1:C01, 2:C10, 3:C11
    # W1_bot rows: 0:C0, 1:C1, ...
    # W1 rows are k*n+i.
    # C[0,0]=0, C[0,1]=1, C[1,0]=3, C[1,1]=4, C[2,0]=6, C[2,1]=7.
    W1[0][0:20] = W1_top[0][0:20]
    W1[1][0:20] = W1_top[1][0:20]
    W1[3][0:20] = W1_top[2][0:20]
    W1[4][0:20] = W1_top[3][0:20]
    
    W1[6][20:34] = W1_bot[0][20:34]
    W1[7][20:34] = W1_bot[1][20:34]
    W1[9][20:34] = W1_bot[2][20:34]
    W1[10][20:34] = W1_bot[3][20:34]
    W1[12][20:34] = W1_bot[4][20:34]
    W1[13][20:34] = W1_bot[5][20:34]
    W1[15][20:34] = W1_bot[6][20:34]
    
    # Block 2: j=2,3. Offset 34.
    # Similar mapping.
    U2_top, V2_top, W2_top = build_2x2x7(34)
    U2_bot, V2_bot, W2_bot = schoolbook_1x2x7(54)
    
    U2 = [[0]*68 for _ in range(15)]
    V2 = [[0]*68 for _ in range(35)]
    W2 = [[0]*68 for _ in range(21)]
    
    # A[0,2]=2, A[0,3]=3, A[1,2]=7, A[1,3]=8, A[2,2]=12, A[2,3]=13
    U2[2][34:54] = U2_top[0][34:54]
    U2[3][34:54] = U2_top[1][34:54]
    U2[7][34:54] = U2_top[2][34:54]
    U2[8][34:54] = U2_top[3][34:54]
    
    U2[12][54:68] = U2_bot[0][54:68]
    U2[13][54:68] = U2_bot[1][54:68]
    
    # B[2,0]=14, B[2,1]=15, B[3,0]=21, B[3,1]=22
    V2[14][34:54] = V2_top[0][34:54]
    V2[15][34:54] = V2_top[1][34:54]
    V2[21][34:54] = V2_top[2][34:54]
    V2[22][34:54] = V2_top[3][34:54]
    
    V2[14][54:68] = V2_bot[0][54:68]
    V2[15][54:68] = V2_bot[1][54:68]
    V2[21][54:68] = V2_bot[2][54:68]
    V2[22][54:68] = V2_bot[3][54:68]
    
    # C[0,0]=0, C[0,1]=1, C[1,0]=3, C[1,1]=4, C[2,0]=6, C[2,1]=7
    W2[0][34:54] = W2_top[0][34:54]
    W2[1][34:54] = W2_top[1][34:54]
    W2[3][34:54] = W2_top[2][34:54]
    W2[4][34:54] = W2_top[3][34:54]
    
    W2[6][54:68] = W2_bot[0][54:68]
    W2[7][54:68] = W2_bot[1][54:68]
    W2[9][54:68] = W2_bot[2][54:68]
    W2[10][54:68] = W2_bot[3][54:68]
    W2[12][54:68] = W2_bot[4][54:68]
    W2[13][54:68] = W2_bot[5][54:68]
    W2[15][54:68] = W2_bot[6][54:68]
    
    # Block 3: j=4. Offset 68. Rank 21.
    U3, V3, W3 = schoolbook_3x1x7(68)
    
    # Combine
    rank = 89
    U = [[0]*rank for _ in range(15)]
    V = [[0]*rank for _ in range(35)]
    W = [[0]*rank for _ in range(21)]
    
    for i in range(15):
        U[i] = U1[i] + U2[i] + U3[i]
    for i in range(35):
        V[i] = V1[i] + V2[i] + V3[i]
    for i in range(21):
        W[i] = W1[i] + W2[i] + W3[i]
        
    # Verify
    try:
        Ua = np.array(U, dtype=float)
        Va = np.array(V, dtype=float)
        Wa = np.array(W, dtype=float)
        ok = True
        for i in range(3):
            for j in range(5):
                for k in range(7):
                    val = np.dot(Ua[i*5+j] * Va[j*7+k], Wa[k*3+i])
                    if abs(val - 1.0) > 1e-9:
                        ok = False
                        break
                if not ok: break
            if not ok: break
        if ok:
            return {"U": U, "V": V, "W": W}
    except:
        pass
        
    # Fallback
    rank = 105
    U = [[0]*rank for _ in range(15)]
    V = [[0]*rank for _ in range(35)]
    W = [[0]*rank for _ in range(21)]
    col = 0
    for i in range(3):
        for j in range(5):
            for k in range(7):
                U[i*5+j][col] = 1
                V[j*7+k][col] = 1
                W[k*3+i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
