import numpy as np

DIMENSIONS = [2, 4, 6]

def construct():
    n, m, p = DIMENSIONS
    # We split A into 2 blocks of 2x2 (A_L, A_R)
    # We split B into 2 blocks of 2x6 (B_T, B_B)
    # B_T is split into 3 blocks of 2x2 (B_T1, B_T2, B_T3)
    # B_B is split into 3 blocks of 2x2 (B_B1, B_B2, B_B3)
    
    # Total 6 products of 2x2 by 2x2
    # Each uses Strassen (rank 7)
    # Total rank 42
    
    rank = 42
    U = [[0] * rank for _ in range(n * m)] # 8 x 42
    V = [[0] * rank for _ in range(m * p)] # 24 x 42
    W = [[0] * rank for _ in range(p * n)] # 12 x 42
    
    col = 0
    
    # Helper to add a Strassen block
    # a_idx, b_idx are indices for A and B blocks
    # a_block: 2x2 matrix of indices in U
    # b_block: 2x2 matrix of indices in V
    # c_block: 2x2 matrix of indices in W
    def add_strassen(a_idx, b_idx, c_idx):
        nonlocal col
        # a_idx[i][j] is the row index in U for A[i][j]
        # b_idx[i][j] is the row index in V for B[i][j]
        # c_idx[i][j] is the row index in W for C[i][j]
        
        # M1 = (a00 + a11)(b00 + b11)
        U[a_idx[0][0]][col] += 1
        U[a_idx[1][1]][col] += 1
        V[b_idx[0][0]][col] += 1
        V[b_idx[1][1]][col] += 1
        W[c_idx[0][0]][col] += 1
        W[c_idx[1][1]][col] += 1
        W[c_idx[0][1]][col] -= 1
        W[c_idx[1][0]][col] += 1
        col += 1
        
        # M2 = (a10 + a11)b00
        U[a_idx[1][0]][col] += 1
        U[a_idx[1][1]][col] += 1
        V[b_idx[0][0]][col] += 1
        W[c_idx[1][0]][col] += 1
        W[c_idx[1][1]][col] -= 1
        col += 1
        
        # M3 = a00(b01 - b11)
        U[a_idx[0][0]][col] += 1
        V[b_idx[0][1]][col] += 1
        V[b_idx[1][1]][col] -= 1
        W[c_idx[0][1]][col] += 1
        W[c_idx[1][1]][col] += 1
        col += 1
        
        # M4 = a11(b10 - b00)
        U[a_idx[1][1]][col] += 1
        V[b_idx[1][0]][col] += 1
        V[b_idx[0][0]][col] -= 1
        W[c_idx[0][0]][col] += 1
        W[c_idx[1][0]][col] += 1
        col += 1
        
        # M5 = (a00 + a01)b11
        U[a_idx[0][0]][col] += 1
        U[a_idx[0][1]][col] += 1
        V[b_idx[1][1]][col] += 1
        W[c_idx[0][0]][col] -= 1
        W[c_idx[0][1]][col] += 1
        col += 1
        
        # M6 = (a10 - a00)(b00 + b01)
        U[a_idx[1][0]][col] += 1
        U[a_idx[0][0]][col] -= 1
        V[b_idx[0][0]][col] += 1
        V[b_idx[0][1]][col] += 1
        W[c_idx[1][1]][col] += 1
        col += 1
        
        # M7 = (a01 - a11)(b10 + b11)
        U[a_idx[0][1]][col] += 1
        U[a_idx[1][1]][col] -= 1
        V[b_idx[1][0]][col] += 1
        V[b_idx[1][1]][col] += 1
        W[c_idx[0][0]][col] += 1
        W[c_idx[1][1]][col] += 1
        col += 1

    # A is 2x4. Indices:
    # A[0,0] -> U[0], A[0,1] -> U[1], A[0,2] -> U[2], A[0,3] -> U[3]
    # A[1,0] -> U[4], A[1,1] -> U[5], A[1,2] -> U[6], A[1,3] -> U[7]
    
    # B is 4x6. Indices:
    # B[0,0] -> V[0], ..., B[0,5] -> V[5]
    # B[1,0] -> V[6], ..., B[1,5] -> V[11]
    # B[2,0] -> V[12], ..., B[2,5] -> V[17]
    # B[3,0] -> V[18], ..., B[3,5] -> V[23]
    
    # C is 2x6. Indices in W (trace convention W[k*n+i, r]):
    # C[0,0] -> W[0], C[0,1] -> W[1], ..., C[0,5] -> W[5]
    # C[1,0] -> W[6], C[1,1] -> W[7], ..., C[1,5] -> W[11]
    
    # Block 1: A_L (rows 0-1, cols 0-1) * B_T1 (rows 0-1, cols 0-1)
    # A_L indices: [[0, 1], [4, 5]]
    # B_T1 indices: [[0, 1], [6, 7]]
    # C_11 indices: [[0, 1], [6, 7]]
    a1 = [[0, 1], [4, 5]]
    b1 = [[0, 1], [6, 7]]
    c1 = [[0, 1], [6, 7]]
    add_strassen(a1, b1, c1)
    
    # Block 2: A_L * B_T2 (rows 0-1, cols 2-3)
    # B_T2 indices: [[2, 3], [8, 9]]
    # C_12 indices: [[2, 3], [8, 9]]
    b2 = [[2, 3], [8, 9]]
    c2 = [[2, 3], [8, 9]]
    add_strassen(a1, b2, c2)
    
    # Block 3: A_L * B_T3 (rows 0-1, cols 4-5)
    # B_T3 indices: [[4, 5], [10, 11]]
    # C_13 indices: [[4, 5], [10, 11]]
    b3 = [[4, 5], [10, 11]]
    c3 = [[4, 5], [10, 11]]
    add_strassen(a1, b3, c3)
    
    # Block 4: A_R (rows 0-1, cols 2-3) * B_B1 (rows 2-3, cols 0-1)
    # A_R indices: [[2, 3], [6, 7]]
    # B_B1 indices: [[12, 13], [18, 19]]
    # C_21 indices: [[0, 1], [6, 7]]
    a2 = [[2, 3], [6, 7]]
    b4 = [[12, 13], [18, 19]]
    c4 = [[0, 1], [6, 7]]
    add_strassen(a2, b4, c4)
    
    # Block 5: A_R * B_B2 (rows 2-3, cols 2-3)
    # B_B2 indices: [[14, 15], [20, 21]]
    # C_22 indices: [[2, 3], [8, 9]]
    b5 = [[14, 15], [20, 21]]
    c5 = [[2, 3], [8, 9]]
    add_strassen(a2, b5, c5)
    
    # Block 6: A_R * B_B3 (rows 2-3, cols 4-5)
    # B_B3 indices: [[16, 17], [22, 23]]
    # C_23 indices: [[4, 5], [10, 11]]
    b6 = [[16, 17], [22, 23]]
    c6 = [[4, 5], [10, 11]]
    add_strassen(a2, b6, c6)
    
    return {"U": U, "V": V, "W": W}
