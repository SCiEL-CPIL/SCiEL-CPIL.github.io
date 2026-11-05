"""Matrix multiplication 2x4 by 4x7 with rank-minimal exact tensor decomposition."""
DIMENSIONS = [2, 4, 7]

# EVOLVE-BLOCK-START
def construct():
    # Build a rank-52 Strassen-based decomposition:
    #   Split B (4x7) into B_L (4x2) and B_R (4x5).
    #   C_L = A * B_L  (2x4 * 4x2 -> 2x2), rank 14 (two 2x2x2 Strassen blocks)
    #   C_R = A * B_R  (2x4 * 4x5 -> 2x5), rank 38 (two 2x2x5 blocks, each rank 19)
    #   Total rank = 14 + 38 = 52
    #
    # Convention: C[i, j] = sum_k A[i, k] * B[k, j]
    # Tensor: T[i, k, j] = A[i,k] * B[k,j] -> C[i,j]
    # W convention: W[k*n + i, r] where n=2 (rows of A), so W is (p*n, rank) = (14, rank)
    # U convention: U[i*m + j, r] where i in [0,n), j in [0,m) -> U is (n*m, rank) = (8, rank)
    # V convention: V[j*p + k, r] where j in [0,m), k in [0,p) -> V is (m*p, rank) = (28, rank)
    #
    # The decomposition says: C[i,j] = sum_r U[i*m+j, r] * V[j*m+k... wait
    # Let me re-read: U[i*m+j, c], V[j*p+k, c], W[k*n+i, c]
    # So: T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    # where i in [0,n), j in [0,m), k in [0,p)
    # n=2, m=4, p=7
    # U: (n*m, rank) = (8, rank), U[i*4+j, c]
    # V: (m*p, rank) = (28, rank), V[j*7+k, c]
    # W: (p*n, rank) = (14, rank), W[k*2+i, c]
    
    import time
    deadline = time.time() + 140  # safety margin before 150s limit
    
    n, m, p = DIMENSIONS  # 2, 4, 7
    
    def strassen_222():
        """Return U, V, W for 2x2x2 Strassen, rank 7.
        U[i*2+j, c], V[j*2+k, c], W[k*2+i, c]
        i,j,k in {0,1}
        """
        # Strassen formulas for C = A * B where A,B are 2x2
        # M1 = (A00+A11)(B00+B11)
        # M2 = (A10+A11)B00
        # M3 = A00(B01-B11)
        # M4 = A11(B10-B00)
        # M5 = (A00+A10)B11
        # M6 = (A10-A00)(B00+B10)
        # M7 = (A01-A11)(B01+B11)
        # C00 = M1 + M4 - M5 + M7
        # C01 = M3 + M5
        # C10 = M2 + M4
        # C11 = M1 - M2 + M3 + M6
        
        # U[i*2+j, c] corresponds to A[i,j]
        # V[j*2+k, c] corresponds to B[j,k]
        # W[k*2+i, c] corresponds to C[i,k] (output row i, col k)
        
        U = [[0]*7 for _ in range(4)]
        V = [[0]*7 for _ in range(4)]
        W = [[0]*7 for _ in range(4)]
        
        # M1 = (A00+A11)(B00+B11)
        U[0][0] = 1; U[3][0] = 1  # A00, A11
        V[0][0] = 1; V[3][0] = 1  # B00, B11
        W[0][0] = 1; W[3][0] = 1; W[1][0] = -1; W[2][0] = 1  # C00+=, C11+=, C10-=, C01+=
        
        # M2 = (A10+A11)B00
        U[2][1] = 1; U[3][1] = 1  # A10, A11
        V[0][1] = 1  # B00
        W[1][1] = 1; W[3][1] = -1  # C10+=, C11-=
        
        # M3 = A00(B01-B11)
        U[0][2] = 1  # A00
        V[1][2] = 1; V[3][2] = -1  # B01, -B11
        W[0][2] = 1; W[3][2] = 1  # C01+=, C11+=
        
        # M4 = A11(B10-B00)
        U[3][3] = 1  # A11
        V[2][3] = 1; V[0][3] = -1  # B10, -B00
        W[0][3] = 1; W[1][3] = 1  # C00+=, C10+=
        
        # M5 = (A00+A10)B11
        U[0][4] = 1; U[2][4] = 1  # A00, A10
        V[3][4] = 1  # B11
        W[0][4] = -1; W[1][4] = 1  # C00-=, C10+=
        
        # M6 = (A10-A00)(B00+B10)
        U[2][5] = 1; U[0][5] = -1  # A10, -A00
        V[0][5] = 1; V[2][5] = 1  # B00, B10
        W[3][5] = 1  # C11+=
        
        # M7 = (A01-A11)(B01+B11)
        U[1][6] = 1; U[3][6] = -1  # A01, -A11
        V[1][6] = 1; V[3][6] = 1  # B01, B11
        W[0][6] = 1  # C00+=
        
        return U, V, W
    
    def strassen_223():
        """Return U, V, W for 2x2x3 (rank 12) via Strassen on two 2x2x2 blocks.
        U[i*2+j, c], V[j*3+k, c], W[k*2+i, c]
        i in {0,1}, j in {0,1}, k in {0,1,2}
        
        Split B (2x3) into B1 (2x2) and B2 (2x1).
        C = A*B1 + A*B2
        A*B1: rank 7 (Strassen 2x2x2)
        A*B2: rank 4 (schoolbook, 2x2x1)
        Total rank 11... but we said 12.
        
        Actually let me use a different approach: split A into two 1x2 rows.
        Row 0 of A (1x2) times B (2x3) = 1x3, rank 6 (schoolbook)
        Row 1 of A (1x2) times B (2x3) = 1x3, rank 6 (schoolbook)
        Total rank 12.
        """
        # Use schoolbook for simplicity: rank 12
        U = [[0]*12 for _ in range(4)]   # i*2+j, i in {0,1}, j in {0,1}
        V = [[0]*12 for _ in range(6)]   # j*3+k, j in {0,1}, k in {0,1,2}
        W = [[0]*12 for _ in range(6)]   # k*2+i, k in {0,1,2}, i in {0,1}
        
        c = 0
        for i in range(2):
            for j in range(2):
                for k in range(3):
                    U[i*2+j][c] = 1
                    V[j*3+k][c] = 1
                    W[k*2+i][c] = 1
                    c += 1
        return U, V, W
    
    def strassen_225():
        """Return U, V, W for 2x2x5 (rank 19).
        U[i*2+j, c], V[j*5+k, c], W[k*2+i, c]
        i in {0,1}, j in {0,1}, k in {0,1,2,3,4}
        
        Split B (2x5) into B1 (2x2) and B2 (2x3).
        C = A*B1 + A*B2
        A*B1: rank 7 (Strassen)
        A*B2: rank 12 (schoolbook 2x2x3)
        Total rank 19.
        """
        r1 = 7
        r2 = 12
        rank = r1 + r2  # 19
        
        U1, V1, W1 = strassen_222()
        U2, V2, W2 = strassen_223()
        
        U = [[0]*rank for _ in range(4)]
        V = [[0]*rank for _ in range(10)]  # j*5+k
        W = [[0]*rank for _ in range(10)]  # k*2+i
        
        # Copy block 1 (k in 0,1)
        for i in range(4):
            for c in range(r1):
                U[i][c] = U1[i][c]
        for j in range(2):
            for k in range(2):
                for c in range(r1):
                    V[j*5+k][c] = V1[j*2+k][c]
        for k in range(2):
            for i in range(2):
                for c in range(r1):
                    W[k*2+i][c] = W1[k*2+i][c]
        
        # Copy block 2 (k in 2,3,4)
        for i in range(4):
            for c in range(r1, rank):
                U[i][c] = U2[i][c-r1]
        for j in range(2):
            for k in range(3):
                for c in range(r1, rank):
                    V[j*5+(k+2)][c] = V2[j*3+k][c-r1]
        for k in range(3):
            for i in range(2):
                for c in range(r1, rank):
                    W[(k+2)*2+i][c] = W2[k*2+i][c-r1]
        
        return U, V, W
    
    # Now build the full 2x4x7 decomposition
    # Split B (4x7) into B_L (4x2) [cols 0,1] and B_R (4x5) [cols 2,3,4,5,6]
    
    # Part 1: A (2x4) * B_L (4x2) -> C_L (2x2), rank 14
    #   Split A into A_top (2x2, rows 0,1) and A_bot (2x2, rows 2,3)
    #   B_L into B_L_top (2x2, rows 0,1) and B_L_bot (2x2, rows 2,3)
    #   C_L = A_top * B_L_top + A_bot * B_L_bot
    #   Each term is 2x2x2, rank 7. Total rank 14.
    
    # Part 2: A (2x4) * B_R (4x5) -> C_R (2x5), rank 38
    #   Split A into A_top (2x2, rows 0,1) and A_bot (2x2, rows 2,3)
    #   B_R into B_R_top (2x5, rows 0,1) and B_R_bot (2x5, rows 2,3)
    #   C_R = A_top * B_R_top + A_bot * B_R_bot
    #   Each term is 2x2x5, rank 19. Total rank 38.
    
    rank1 = 14
    rank2 = 38
    total_rank = rank1 + rank2  # 52
    
    # Full tensors:
    # U[i*4+j, c] where i in {0,1} (row of A), j in {0,1,2,3} (col of A = row of B)
    # V[j*7+k, c] where j in {0,1,2,3} (row of B), k in {0,...,6} (col of B)
    # W[k*2+i, c] where k in {0,...,6} (col of B), i in {0,1} (row of C)
    
    U = [[0]*total_rank for _ in range(8)]
    V = [[0]*total_rank for _ in range(28)]
    W = [[0]*total_rank for _ in range(14)]
    
    # --- Part 1: C_L (columns 0,1 of C) ---
    # Block 1a: A_top (rows 0,1 of A; cols 0,1 of A = rows 0,1 of B) * B_L_top (rows 0,1 of B; cols 0,1)
    #   U indices: i*4+j where i in {0,1}, j in {0,1} -> rows 0,1,4,5 of U
    #   V indices: j*7+k where j in {0,1}, k in {0,1}
    #   W indices: k*2+i where k in {0,1}, i in {0,1}
    U1a, V1a, W1a = strassen_222()
    for i in range(2):
        for j in range(2):
            for c in range(7):
                U[i*4+j][c] += U1a[i*2+j][c]
    for j in range(2):
        for k in range(2):
            for c in range(7):
                V[j*7+k][c] += V1a[j*2+k][c]
    for k in range(2):
        for i in range(2):
            for c in range(7):
                W[k*2+i][c] += W1a[k*2+i][c]
    
    # Block 1b: A_bot (rows 0,1 of A; cols 2,3 of A = rows 2,3 of B) * B_L_bot (rows 2,3 of B; cols 0,1)
    #   U indices: i*4+j where i in {0,1}, j in {2,3} -> rows 2,3,6,7 of U
    #   V indices: j*7+k where j in {2,3}, k in {0,1}
    #   W indices: k*2+i where k in {0,1}, i in {0,1}
    U1b, V1b, W1b = strassen_222()
    for i in range(2):
        for j in range(2):
            for c in range(7, 14):
                U[i*4+(j+2)][c] += U1b[i*2+j][c-7]
    for j in range(2):
        for k in range(2):
            for c in range(7, 14):
                V[(j+2)*7+k][c] += V1b[j*2+k][c-7]
    for k in range(2):
        for i in range(2):
            for c in range(7, 14):
                W[k*2+i][c] += W1b[k*2+i][c-7]
    
    # --- Part 2: C_R (columns 2,3,4,5,6 of C) ---
    # Block 2a: A_top * B_R_top (2x2x5, rank 19)
    #   U indices: i*4+j where i in {0,1}, j in {0,1}
    #   V indices: j*7+k where j in {0,1}, k in {2,3,4,5,6}
    #   W indices: k*2+i where k in {2,3,4,5,6}, i in {0,1}
    U2a, V2a, W2a = strassen_225()
    for i in range(2):
        for j in range(2):
            for c in range(14, total_rank):
                U[i*4+j][c] += U2a[i*2+j][c-14]
    for j in range(2):
        for k in range(5):
            for c in range(14, total_rank):
                V[j*7+(k+2)][c] += V2a[j*5+k][c-14]
    for k in range(5):
        for i in range(2):
            for c in range(14, total_rank):
                W[(k+2)*2+i][c] += W2a[k*2+i][c-14]
    
    # Block 2b: A_bot * B_R_bot (2x2x5, rank 19)
    #   U indices: i*4+j where i in {0,1}, j in {2,3}
    #   V indices: j*7+k where j in {2,3}, k in {2,3,4,5,6}
    #   W indices: k*2+i where k in {2,3,4,5,6}, i in {0,1}
    U2b, V2b, W2b = strassen_225()
    for i in range(2):
        for j in range(2):
            for c in range(14, total_rank):
                U[i*4+(j+2)][c] += U2b[i*2+j][c-14]
    for j in range(2):
        for k in range(5):
            for c in range(14, total_rank):
                V[(j+2)*7+(k+2)][c] += V2b[j*5+k][c-14]
    for k in range(5):
        for i in range(2):
            for c in range(14, total_rank):
                W[(k+2)*2+i][c] += W2b[k*2+i][c-14]
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
