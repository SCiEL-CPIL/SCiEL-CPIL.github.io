import numpy as np
import time

DIMENSIONS = [2, 4, 6]

def construct():
    n, m, p = DIMENSIONS
    
    # Rank 42 decomposition based on Strassen's algorithm applied to 2x2 blocks
    # M_2,4,6 decomposed into 3 blocks of M_2,4,2, each using 14 factors (2x Strassen 2x2x2)
    rank = 42
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Helper to add a Strassen 2x2x2 decomposition
    # a_idx: indices of A block (4 elements: [a00, a01, a10, a11])
    # b_idx: indices of B block (4 elements: [b00, b01, b10, b11])
    # w_idx: indices of C block (4 elements: [c00, c01, c10, c11])
    # start_col: starting column index for this block's factors
    def add_strassen_block(a_idx, b_idx, w_idx, start_col):
        # Strassen formulas:
        # M1 = (a00+a11)(b00+b11)
        # M2 = (a10+a11)b00
        # M3 = a00(b01-b11)
        # M4 = a11(b10-b00)
        # M5 = (a00+a01)b11
        # M6 = (a10-a00)(b00+b01)
        # M7 = (a01-a11)(b10+b11)
        
        # U (A) contributions
        # M1: a00, a11
        U[a_idx[0]][start_col + 0] += 1
        U[a_idx[3]][start_col + 0] += 1
        # M2: a10, a11
        U[a_idx[2]][start_col + 1] += 1
        U[a_idx[3]][start_col + 1] += 1
        # M3: a00
        U[a_idx[0]][start_col + 2] += 1
        # M4: a11
        U[a_idx[3]][start_col + 3] += 1
        # M5: a00, a01
        U[a_idx[0]][start_col + 4] += 1
        U[a_idx[1]][start_col + 4] += 1
        # M6: a10, -a00
        U[a_idx[2]][start_col + 5] += 1
        U[a_idx[0]][start_col + 5] -= 1
        # M7: a01, -a11
        U[a_idx[1]][start_col + 6] += 1
        U[a_idx[3]][start_col + 6] -= 1
        
        # V (B) contributions
        # M1: b00, b11
        V[b_idx[0]][start_col + 0] += 1
        V[b_idx[3]][start_col + 0] += 1
        # M2: b00
        V[b_idx[0]][start_col + 1] += 1
        # M3: b01, -b11
        V[b_idx[1]][start_col + 2] += 1
        V[b_idx[3]][start_col + 2] -= 1
        # M4: b10, -b00
        V[b_idx[2]][start_col + 3] += 1
        V[b_idx[0]][start_col + 3] -= 1
        # M5: b11
        V[b_idx[3]][start_col + 4] += 1
        # M6: b00, b01
        V[b_idx[0]][start_col + 5] += 1
        V[b_idx[1]][start_col + 5] += 1
        # M7: b10, b11
        V[b_idx[2]][start_col + 6] += 1
        V[b_idx[3]][start_col + 6] += 1
        
        # W (C) contributions
        # z00 = M1 + M4 - M5 + M7
        W[w_idx[0]][start_col + 0] += 1
        W[w_idx[0]][start_col + 3] += 1
        W[w_idx[0]][start_col + 4] -= 1
        W[w_idx[0]][start_col + 6] += 1
        
        # z01 = M3 + M5
        W[w_idx[1]][start_col + 2] += 1
        W[w_idx[1]][start_col + 4] += 1
        
        # z10 = M2 + M4
        W[w_idx[2]][start_col + 1] += 1
        W[w_idx[2]][start_col + 3] += 1
        
        # z11 = M1 - M2 + M3 + M6
        W[w_idx[3]][start_col + 0] += 1
        W[w_idx[3]][start_col + 1] -= 1
        W[w_idx[3]][start_col + 2] += 1
        W[w_idx[3]][start_col + 5] += 1

    # A is 2x4. Indices:
    # A[0,0]=0, A[0,1]=1, A[0,2]=2, A[0,3]=3
    # A[1,0]=4, A[1,1]=5, A[1,2]=6, A[1,3]=7
    # A_L = [0, 1, 4, 5] (a00, a01, a10, a11)
    # A_R = [2, 3, 6, 7]
    
    a_L = [0, 1, 4, 5]
    a_R = [2, 3, 6, 7]
    
    # B is 4x6. Indices j*p + k
    # j=0: 0..5, j=1: 6..11, j=2: 12..17, j=3: 18..23
    
    # Block 1: B cols 0,1
    # B_T (rows 0,1): 
    # b00=B[0,0]=0, b01=B[0,1]=1, b10=B[1,0]=6, b11=B[1,1]=7
    # B_B (rows 2,3):
    # b00=B[2,0]=12, b01=B[2,1]=13, b10=B[3,0]=18, b11=B[3,1]=19
    
    b_1_T = [0, 1, 6, 7]
    b_1_B = [12, 13, 18, 19]
    
    # W indices for C cols 0,1
    # C is 2x6. W index k*n + i.
    # k is col index of C. i is row index of C.
    # C[0,0] -> k=0, i=0 -> W[0*2+0] = W[0]
    # C[0,1] -> k=1, i=0 -> W[1*2+0] = W[2]
    # C[1,0] -> k=0, i=1 -> W[0*2+1] = W[1]
    # C[1,1] -> k=1, i=1 -> W[1*2+1] = W[3]
    # w_idx for block 1 is [0, 2, 1, 3] corresponding to z00, z01, z10, z11.
    
    w_1 = [0, 2, 1, 3]
    
    add_strassen_block(a_L, b_1_T, w_1, 0)
    add_strassen_block(a_R, b_1_B, w_1, 7)
    
    # Block 2: B cols 2,3
    # B_T:
    # b00=B[0,2]=2, b01=B[0,3]=3, b10=B[1,2]=8, b11=B[1,3]=9
    # B_B:
    # b00=B[2,2]=14, b01=B[2,3]=15, b10=B[3,2]=20, b11=B[3,3]=21
    
    b_2_T = [2, 3, 8, 9]
    b_2_B = [14, 15, 20, 21]
    
    # W indices for C cols 2,3
    # C[0,2] -> k=2, i=0 -> W[4]
    # C[0,3] -> k=3, i=0 -> W[6]
    # C[1,2] -> k=2, i=1 -> W[5]
    # C[1,3] -> k=3, i=1 -> W[7]
    
    w_2 = [4, 6, 5, 7]
    
    add_strassen_block(a_L, b_2_T, w_2, 14)
    add_strassen_block(a_R, b_2_B, w_2, 21)
    
    # Block 3: B cols 4,5
    # B_T:
    # b00=B[0,4]=4, b01=B[0,5]=5, b10=B[1,4]=10, b11=B[1,5]=11
    # B_B:
    # b00=B[2,4]=16, b01=B[2,5]=17, b10=B[3,4]=22, b11=B[3,5]=23
    
    b_3_T = [4, 5, 10, 11]
    b_3_B = [16, 17, 22, 23]
    
    # W indices for C cols 4,5
    # C[0,4] -> k=4, i=0 -> W[8]
    # C[0,5] -> k=5, i=0 -> W[10]
    # C[1,4] -> k=4, i=1 -> W[9]
    # C[1,5] -> k=5, i=1 -> W[11]
    
    w_3 = [8, 10, 9, 11]
    
    add_strassen_block(a_L, b_3_T, w_3, 28)
    add_strassen_block(a_R, b_3_B, w_3, 35)
    
    return {"U": U, "V": V, "W": W}
