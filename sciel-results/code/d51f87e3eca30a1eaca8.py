import numpy as np

def construct():
    n, m, p = 3, 3, 7
    rank = 60
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    col = 0
    
    # Helper to add a trivial 2x2x1 block
    def add_trivial_2x2x1(a_idx, b_idx, c_idx):
        nonlocal col
        # a_idx: 4 indices for 2x2 A
        # b_idx: 2 indices for 2x1 B
        # c_idx: 2 indices for 2x1 C
        # C0 = A00*B0 + A01*B1
        # C1 = A10*B0 + A11*B1
        U[a_idx[0], col] = 1; V[b_idx[0], col] = 1; W[c_idx[0], col] = 1; col += 1
        U[a_idx[1], col] = 1; V[b_idx[1], col] = 1; W[c_idx[0], col] = 1; col += 1
        U[a_idx[2], col] = 1; V[b_idx[0], col] = 1; W[c_idx[1], col] = 1; col += 1
        U[a_idx[3], col] = 1; V[b_idx[1], col] = 1; W[c_idx[1], col] = 1; col += 1

    # Helper to add a Strassen 2x2x2 block
    def add_strassen_2x2x2(a_idx, b_idx, c_idx):
        nonlocal col
        # a_idx: 4 indices for 2x2 A (00, 01, 10, 11)
        # b_idx: 4 indices for 2x2 B (00, 01, 10, 11)
        # c_idx: 4 indices for 2x2 C (00, 01, 10, 11)
        
        # M1 = (A00+A11)(B00+B11)
        U[a_idx[0], col] = 1; U[a_idx[3], col] = 1
        V[b_idx[0], col] = 1; V[b_idx[3], col] = 1
        W[c_idx[0], col] = 1; W[c_idx[3], col] = 1
        col += 1
        
        # M2 = (A10+A11)B00
        U[a_idx[2], col] = 1; U[a_idx[3], col] = 1
        V[b_idx[0], col] = 1
        W[c_idx[2], col] = 1; W[c_idx[3], col] = 1
        col += 1
        
        # M3 = A00(B01-B11)
        U[a_idx[0], col] = 1
        V[b_idx[1], col] = 1; V[b_idx[3], col] = -1
        W[c_idx[1], col] = 1; W[c_idx[3], col] = 1
        col += 1
        
        # M4 = A11(B10-B00)
        U[a_idx[3], col] = 1
        V[b_idx[2], col] = 1; V[b_idx[0], col] = -1
        W[c_idx[0], col] = 1; W[c_idx[3], col] = 1
        col += 1
        
        # M5 = (A00+A10)B11
        U[a_idx[0], col] = 1; U[a_idx[2], col] = 1
        V[b_idx[3], col] = 1
        W[c_idx[0], col] = -1; W[c_idx[2], col] = 1
        col += 1
        
        # M6 = (A01-A00)(B00+B10)
        U[a_idx[1], col] = 1; U[a_idx[0], col] = -1
        V[b_idx[0], col] = 1; V[b_idx[2], col] = 1
        W[c_idx[1], col] = 1
        col += 1
        
        # M7 = (A11-A10)(B01+B11)
        U[a_idx[3], col] = 1; U[a_idx[2], col] = -1
        V[b_idx[1], col] = 1; V[b_idx[3], col] = 1
        W[c_idx[0], col] = -1; W[c_idx[1], col] = 1
        col += 1

    # Indices for A (3x3)
    # 0 1 2
    # 3 4 5
    # 6 7 8
    
    # Indices for B (3x7)
    # 0 1 2 3 4 5 6
    # 7 8 9 10 11 12 13
    # 14 15 16 17 18 19 20
    
    # Indices for C (3x7)
    # 0 1 2 3 4 5 6
    # 7 8 9 10 11 12 13
    # 14 15 16 17 18 19 20
    
    # Block 1: A11 (0,1,3,4) * B1a (0,1,7,8) -> C11a (0,1,7,8)
    add_strassen_2x2x2([0, 1, 3, 4], [0, 1, 7, 8], [0, 1, 7, 8])
    
    # Block 2: A11 (0,1,3,4) * B1b (2,3,9,10) -> C11b (2,3,9,10)
    add_strassen_2x2x2([0, 1, 3, 4], [2, 3, 9, 10], [2, 3, 9, 10])
    
    # Block 3: A11 (0,1,3,4) * B1c (4,5,11,12) -> C11c (4,5,11,12)
    add_strassen_2x2x2([0, 1, 3, 4], [4, 5, 11, 12], [4, 5, 11, 12])
    
    # Block 4: A11 (0,1,3,4) * B1d (6,13) -> C11d (6,13)
    add_trivial_2x2x1([0, 1, 3, 4], [6, 13], [6, 13])
    
    # Block 5: A12 (2,5) * B2 (14..20) -> C12 (0..6, 7..13)
    # Trivial 2x1x7
    # C[i,k] = A[i,2] * B[2,k]
    for k in range(7):
        U[2, col] = 1; V[14+k, col] = 1; W[k, col] = 1; col += 1
        U[5, col] = 1; V[14+k, col] = 1; W[7+k, col] = 1; col += 1
        
    # Block 6: A21 (6,7) * B1 (0..6, 7..13) -> C21 (14..20)
    # Trivial 1x2x7
    # C[2,k] = A[2,0]*B[0,k] + A[2,1]*B[1,k]
    for k in range(7):
        U[6, col] = 1; V[k, col] = 1; W[14+k, col] = 1; col += 1
        U[7, col] = 1; V[7+k, col] = 1; W[14+k, col] = 1; col += 1
        
    # Block 7: A22 (8) * B2 (14..20) -> C22 (14..20)
    # Trivial 1x1x7
    for k in range(7):
        U[8, col] = 1; V[14+k, col] = 1; W[14+k, col] = 1; col += 1

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
