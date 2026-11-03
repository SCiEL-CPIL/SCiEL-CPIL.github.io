import numpy as np

DIMENSIONS = [2, 4, 6]

def construct():
    n, m, p = DIMENSIONS
    # Rank 42 decomposition using Strassen on 2x2x2 blocks
    # Structure:
    # A is 2x4, B is 4x6, C is 2x6.
    # Split B into 3 blocks of 4x2: B1, B2, B3.
    # C = A * [B1, B2, B3] = [A B1, A B2, A B3].
    # Each A Bi is 2x2.
    # A is 2x4. Split A into A_L (2x2) and A_R (2x2).
    # Bi is 4x2. Split Bi into Bi_T (2x2) and Bi_B (2x2).
    # A Bi = A_L Bi_T + A_R Bi_B.
    # This is sum of two 2x2x2 multiplications.
    # Use Strassen for each 2x2x2.
    
    rank = 42
    U = [[0] * rank for _ in range(n * m)] # 8 x 42
    V = [[0] * rank for _ in range(m * p)] # 24 x 42
    W = [[0] * rank for _ in range(p * n)] # 12 x 42
    
    # Strassen factors for 2x2x2
    # M1 = (A11 + A22)(B11 + B22)
    # M2 = (A21 + A22)B11
    # M3 = A11(B21 - B22)
    # M4 = A22(B12 - B11)
    # M5 = (A11 + A12)B22
    # M6 = (A21 - A11)(B11 + B12)
    # M7 = (A12 - A22)(B21 + B22)
    
    # We have 3 blocks of p (each width 2).
    # For each block k in 0..2:
    #   We have A_L, A_R (same A for all k)
    #   We have Bk_T, Bk_B (different for each k)
    #   Product = A_L Bk_T + A_R Bk_B
    #   This is two Strassen calls.
    #   Call 1: X = A_L, Y = Bk_T
    #   Call 2: X = A_R, Y = Bk_B
    
    col = 0
    # Indices for A: 0..7 (2x4)
    # A_L: rows 0,1; cols 0,1 -> indices 0,1,2,3 in U?
    # U index is i*m + j.
    # A[0,0]=0, A[0,1]=1, A[0,2]=2, A[0,3]=3
    # A[1,0]=4, A[1,1]=5, A[1,2]=6, A[1,3]=7
    # A_L (cols 0,1): 0,1,4,5
    # A_R (cols 2,3): 2,3,6,7
    
    # Indices for B: 0..23 (4x6)
    # B index is j*p + k.
    # For block k (0,1,2), cols are 2k, 2k+1.
    # Bk_T (rows 0,1): j=0,1.
    #   j=0: k=2k, 2k+1 -> indices 0,1, 4,5? No.
    #   B[0, 2k] = 0*6 + 2k = 2k
    #   B[0, 2k+1] = 2k+1
    #   B[1, 2k] = 1*6 + 2k = 6+2k
    #   B[1, 2k+1] = 7+2k
    #   So Bk_T indices: 2k, 2k+1, 6+2k, 7+2k.
    # Bk_B (rows 2,3): j=2,3.
    #   B[2, 2k] = 12+2k
    #   B[2, 2k+1] = 13+2k
    #   B[3, 2k] = 18+2k
    #   B[3, 2k+1] = 19+2k
    #   So Bk_B indices: 12+2k, 13+2k, 18+2k, 19+2k.
    
    # Indices for C: 0..11 (2x6)
    # C index is k*n + i.
    # For block k, cols 2k, 2k+1.
    # C[0, 2k] = 2k*2 + 0 = 4k
    # C[0, 2k+1] = 4k+1
    # C[1, 2k] = 4k+2
    # C[1, 2k+1] = 4k+3
    
    def add_strassen(U, V, W, col, a_idx, b_idx, c_idx):
        # a_idx: 4 indices for A (2x2)
        # b_idx: 4 indices for B (2x2)
        # c_idx: 4 indices for C (2x2)
        # Returns new col
        # M1
        U[a_idx[0]][col] = 1; U[a_idx[3]][col] = 1
        V[b_idx[0]][col] = 1; V[b_idx[3]][col] = 1
        W[c_idx[0]][col] = 1; W[c_idx[1]][col] = 1; W[c_idx[2]][col] = 1; W[c_idx[3]][col] = 1
        col += 1
        # M2
        U[a_idx[2]][col] = 1; U[a_idx[3]][col] = 1
        V[b_idx[0]][col] = 1
        W[c_idx[0]][col] = 1; W[c_idx[2]][col] = 1
        col += 1
        # M3
        U[a_idx[0]][col] = 1
        V[b_idx[2]][col] = 1; V[b_idx[3]][col] = -1
        W[c_idx[1]][col] = 1; W[c_idx[3]][col] = 1
        col += 1
        # M4
        U[a_idx[3]][col] = 1
        V[b_idx[1]][col] = 1; V[b_idx[0]][col] = -1
        W[c_idx[0]][col] = -1; W[c_idx[2]][col] = -1
        col += 1
        # M5
        U[a_idx[0]][col] = 1; U[a_idx[1]][col] = 1
        V[b_idx[3]][col] = 1
        W[c_idx[1]][col] = 1; W[c_idx[3]][col] = 1
        col += 1
        # M6
        U[a_idx[2]][col] = 1; U[a_idx[0]][col] = -1
        V[b_idx[0]][col] = 1; V[b_idx[1]][col] = 1
        W[c_idx[0]][col] = 1; W[c_idx[2]][col] = 1
        col += 1
        # M7
        U[a_idx[1]][col] = 1; U[a_idx[3]][col] = -1
        V[b_idx[2]][col] = 1; V[b_idx[3]][col] = 1
        W[c_idx[1]][col] = -1; W[c_idx[3]][col] = -1
        col += 1
        return col

    a_L = [0, 1, 4, 5]
    a_R = [2, 3, 6, 7]
    
    for k in range(3):
        b_T = [2*k, 2*k+1, 6+2*k, 7+2*k]
        b_B = [12+2*k, 13+2*k, 18+2*k, 19+2*k]
        c_idx = [4*k, 4*k+1, 4*k+2, 4*k+3]
        
        # A_L * B_T
        col = add_strassen(U, V, W, col, a_L, b_T, c_idx)
        # A_R * B_B
        col = add_strassen(U, V, W, col, a_R, b_B, c_idx)
        
    return {"U": U, "V": V, "W": W}
