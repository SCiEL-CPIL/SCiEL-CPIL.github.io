import numpy as np

DIMENSIONS = [4, 4, 7]

def construct():
    n, m, p = DIMENSIONS
    # Rank 100 decomposition
    # 4x4x4 part (rank 56) + 4x4x3 part (rank 44)
    
    # Helper to create a zero matrix
    def zero_mat(r, c):
        return [[0]*c for _ in range(r)]
    
    # Strassen for 2x2x2
    # U: 4x7, V: 4x7, W: 4x7
    # Indices for A: 0,1,2,3 (i*m+j)
    # Indices for B: 0,1,2,3 (j*p+k)
    # Indices for C: 0,1,2,3 (k*n+i)
    
    def get_strassen_222():
        # Strassen coefficients
        # P1 = A11(B22 - B21)
        # P2 = (A11 + A12)B21
        # P3 = (A21 + A22)B11
        # P4 = A22(B12 - B11)
        # P5 = (A11 + A21)B11
        # P6 = (A12 - A21)(B21 + B22)
        # P7 = (A22 - A12)(B11 + B12)
        
        # We need U, V, W such that sum U V W = C
        # U rows: A indices
        # V rows: B indices
        # W rows: C indices
        
        U = [[0]*7 for _ in range(4)]
        V = [[0]*7 for _ in range(4)]
        W = [[0]*7 for _ in range(4)]
        
        # A indices: 00, 01, 10, 11 -> 0, 1, 2, 3
        # B indices: 00, 01, 10, 11 -> 0, 1, 2, 3
        # C indices: 00, 01, 10, 11 -> 0, 1, 2, 3
        
        # P1
        U[0][0] = 1 # A11
        V[3][0] = 1 # B22
        V[2][0] = -1 # B21
        W[0][0] = 1 # C11
        W[2][0] = 1 # C21? No, C11 = P1 + P4 - P7
        # C11 = P1 + P4 - P7
        # C12 = P2 + P4
        # C21 = P3 + P5
        # C22 = P1 - P3 + P5 + P6
        
        # Let's assign W based on the formula
        # C00 (0) = P1 + P4 - P7
        W[0][0] = 1
        W[0][3] = 1
        W[0][6] = -1
        
        # C01 (1) = P2 + P4
        W[1][1] = 1
        W[1][3] = 1
        
        # C10 (2) = P3 + P5
        W[2][2] = 1
        W[2][4] = 1
        
        # C11 (3) = P1 - P3 + P5 + P6
        W[3][0] = 1
        W[3][2] = -1
        W[3][4] = 1
        W[3][5] = 1
        
        # Now fill U and V for each P
        # P1: A11 * (B22 - B21)
        # A11 is index 0. B22 is index 3. B21 is index 2.
        U[0][0] = 1
        V[3][0] = 1
        V[2][0] = -1
        
        # P2: (A11 + A12) * B21
        # A11 is 0, A12 is 1. B21 is 2.
        U[0][1] = 1
        U[1][1] = 1
        V[2][1] = 1
        
        # P3: (A21 + A22) * B11
        # A21 is 2, A22 is 3. B11 is 0.
        U[2][2] = 1
        U[3][2] = 1
        V[0][2] = 1
        
        # P4: A22 * (B12 - B11)
        # A22 is 3. B12 is 1. B11 is 0.
        U[3][3] = 1
        V[1][3] = 1
        V[0][3] = -1
        
        # P5: (A11 + A21) * B11
        # A11 is 0, A21 is 2. B11 is 0.
        U[0][4] = 1
        U[2][4] = 1
        V[0][4] = 1
        
        # P6: (A12 - A21) * (B21 + B22)
        # A12 is 1, A21 is 2. B21 is 2, B22 is 3.
        U[1][5] = 1
        U[2][5] = -1
        V[2][5] = 1
        V[3][5] = 1
        
        # P7: (A22 - A12) * (B11 + B12)
        # A22 is 3, A12 is 1. B11 is 0, B12 is 1.
        U[3][6] = 1
        U[1][6] = -1
        V[0][6] = 1
        V[1][6] = 1
        
        return U, V, W

    def get_schoolbook_221():
        # 2x2x1
        # U: 4x4, V: 2x4, W: 2x4
        U = [[0]*4 for _ in range(4)]
        V = [[0]*4 for _ in range(2)]
        W = [[0]*4 for _ in range(2)]
        
        # A00 B00 -> C00
        U[0][0] = 1
        V[0][0] = 1
        W[0][0] = 1
        
        # A01 B10 -> C01? No, C is 2x1.
        # C0 = A00 B00 + A01 B10
        # C1 = A10 B00 + A11 B10
        
        # C0 (index 0)
        W[0][0] = 1 # from A00 B00
        W[0][1] = 1 # from A01 B10
        
        # C1 (index 1)
        W[1][2] = 1 # from A10 B00
        W[1][3] = 1 # from A11 B10
        
        # U, V
        # P0: A00 B00
        U[0][0] = 1
        V[0][0] = 1
        
        # P1: A01 B10
        U[1][1] = 1
        V[1][1] = 1
        
        # P2: A10 B00
        U[2][2] = 1
        V[0][2] = 1
        
        # P3: A11 B10
        U[3][3] = 1
        V[1][3] = 1
        
        return U, V, W

    # Assemble
    rank_444 = 56
    rank_443 = 44
    total_rank = rank_444 + rank_443
    
    U = [[0]*total_rank for _ in range(16)]
    V = [[0]*total_rank for _ in range(28)]
    W = [[0]*total_rank for _ in range(28)]
    
    col = 0
    
    # 4x4x4 part
    # 8 blocks of 2x2x2
    # Block (bi, bj) where bi, bj in 0..1
    # A block (bi, bj) is A[bi*2:(bi+1)*2, bj*2:(bj+1)*2]
    # B block (bj, bk) is B[bj*2:(bj+1)*2, bk*2:(bk+1)*2]
    # C block (bi, bk) is C[bi*2:(bi+1)*2, bk*2:(bk+1)*2]
    
    # We need to compute 8 products:
    # P(bi, bj, bk) = A_block(bi, bj) * B_block(bj, bk)
    # This contributes to C_block(bi, bk)
    
    for bi in range(2):
        for bk in range(2):
            # C_block(bi, bk) = sum_bj A_block(bi, bj) B_block(bj, bk)
            # We compute each term separately
            for bj in range(2):
                # Get Strassen decomposition for this 2x2x2
                U_s, V_s, W_s = get_strassen_222()
                
                # Map indices
                # A indices: row = bi*2 + i, col = bj*2 + j
                # Global U index: (bi*2 + i)*4 + (bj*2 + j)
                # B indices: row = bj*2 + j, col = bk*2 + k
                # Global V index: (bj*2 + j)*7 + (bk*2 + k)
                # C indices: row = bk*2 + k, col = bi*2 + i
                # Global W index: (bk*2 + k)*4 + (bi*2 + i)
                
                for r in range(4):
                    for c in range(7):
                        if U_s[r][c] != 0:
                            # r is local A index
                            # local A index r corresponds to i = r//2, j = r%2
                            i, j = divmod(r, 2)
                            global_u_idx = (bi*2 + i)*4 + (bj*2 + j)
                            U[global_u_idx][col+c] += U_s[r][c]
                            
                for r in range(4):
                    for c in range(7):
                        if V_s[r][c] != 0:
                            # r is local B index
                            # local B index r corresponds to j = r//2, k = r%2
                            j, k = divmod(r, 2)
                            global_v_idx = (bj*2 + j)*7 + (bk*2 + k)
                            V[global_v_idx][col+c] += V_s[r][c]
                            
                for r in range(4):
                    for c in range(7):
                        if W_s[r][c] != 0:
                            # r is local C index
                            # local C index r corresponds to k = r//2, i = r%2
                            k, i = divmod(r, 2)
                            global_w_idx = (bk*2 + k)*4 + (bi*2 + i)
                            W[global_w_idx][col+c] += W_s[r][c]
                            
                col += 7
                
    # 4x4x3 part
    # Split B into B1 (4x2) and B2 (4x1)? No, B is 4x7.
    # We used 4 columns for 4x4x4.
    # Remaining 3 columns: indices 4, 5, 6.
    # We can treat this as 4x4x3.
    # Split into 4 blocks of 2x2x3?
    # B columns 4,5,6.
    # B1 is B[:, 4:6] (4x2). B2 is B[:, 6:7] (4x1).
    # We can compute A B1 (4x2) and A B2 (4x1).
    # A B1: 4x4x2.
    # A B2: 4x4x1.
    
    # 4x4x2:
    # Split into 4 blocks of 2x2x2?
    # B1 is 4x2.
    # B1 block (bj, bk) where bk is 0,1.
    # This is similar to the 4x4x4 part but with p=2.
    # We need 8 blocks of 2x2x2?
    # A is 4x4. B is 4x2.
    # A_block(bi, bj) * B_block(bj, bk).
    # bi, bj in 0..1. bk in 0..1.
    # 8 products of 2x2x2.
    # Rank 8 * 7 = 56?
    # Wait, 4x4x2 schoolbook is 32.
    # 56 is worse.
    # For 2x2x2, Strassen is 7.
    # For 2x2x1, Schoolbook is 4.
    
    # Let's use the 2x2x3 block method for the remaining 3 columns.
    # B_rem is 4x3.
    # Split B_rem into B_rem1 (4x2) and B_rem2 (4x1).
    # Compute A B_rem1 (4x2) and A B_rem2 (4x1).
    # A B_rem1:
    # Use 4 blocks of 2x2x2?
    # B_rem1 is 4x2.
    # B_rem1 block (bj, bk) where bk=0,1.
    # 8 products of 2x2x2. Rank 56.
    # A B_rem2:
    # B_rem2 is 4x1.
    # 4 blocks of 2x2x1. Rank 4 * 4 = 16.
    # Total 72.
    # Plus 56 from 4x4x4. Total 128. Worse than 112.
    
    # My previous 44 for 4x4x3 was based on 2x2x3 rank 11.
    # 2x2x3 rank 11 = 7 (2x2x2) + 4 (2x2x1).
    # So for 4x4x3, we have 4 blocks of 2x2x3.
    # Rank 4 * 11 = 44.
    
    # So we need to implement 4 blocks of 2x2x3.
    # B_rem columns 4,5,6.
    # B_rem1 is cols 4,5 (2x2 per block).
    # B_rem2 is col 6 (2x1 per block).
    
    for bi in range(2):
        for bj in range(2):
            # A_block(bi, bj) is 2x2.
            # B_block(bj, 4:6) is 2x2.
            # B_block(bj, 6:7) is 2x1.
            
            # 2x2x2 part (cols 4,5)
            U_s, V_s, W_s = get_strassen_222()
            
            for r in range(4):
                for c in range(7):
                    if U_s[r][c] != 0:
                        i, j = divmod(r, 2)
                        global_u_idx = (bi*2 + i)*4 + (bj*2 + j)
                        U[global_u_idx][col+c] += U_s[r][c]
                        
            for r in range(4):
                for c in range(7):
                    if V_s[r][c] != 0:
                        j, k = divmod(r, 2)
                        # B col 4+k
                        global_v_idx = (bj*2 + j)*7 + (4 + k)
                        V[global_v_idx][col+c] += V_s[r][c]
                        
            for r in range(4):
                for c in range(7):
                    if W_s[r][c] != 0:
                        k, i = divmod(r, 2)
                        # C col 4+k
                        global_w_idx = (4 + k)*4 + (bi*2 + i)
                        W[global_w_idx][col+c] += W_s[r][c]
                        
            col += 7
            
            # 2x2x1 part (col 6)
            U_sb, V_sb, W_sb = get_schoolbook_221()
            
            for r in range(4):
                for c in range(4):
                    if U_sb[r][c] != 0:
                        i, j = divmod(r, 2)
                        global_u_idx = (bi*2 + i)*4 + (bj*2 + j)
                        U[global_u_idx][col+c] += U_sb[r][c]
                        
            for r in range(2):
                for c in range(4):
                    if V_sb[r][c] != 0:
                        j, k = divmod(r, 1) # k is 0
                        # B col 6
                        global_v_idx = (bj*2 + j)*7 + 6
                        V[global_v_idx][col+c] += V_sb[r][c]
                        
            for r in range(2):
                for c in range(4):
                    if W_sb[r][c] != 0:
                        k, i = divmod(r, 1) # k is 0
                        # C col 6
                        global_w_idx = 6*4 + (bi*2 + i)
                        W[global_w_idx][col+c] += W_sb[r][c]
                        
            col += 4
            
    return {"U": U, "V": V, "W": W}
