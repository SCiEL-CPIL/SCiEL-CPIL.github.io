import numpy as np

DIMENSIONS = [4, 4, 5]

def construct():
    n, m, p = DIMENSIONS
    
    # Lander-Bailey rank-19 decomposition for 4x4x4
    # U: (16, 19), V: (16, 19), W: (16, 19)
    # Row indices: U[i*4+j], V[j*4+k], W[k*4+i]
    
    # The Lander-Bailey decomposition coefficients (from known literature)
    # This is a well-known rank-19 decomposition for 4x4x4 matrix multiplication.
    
    # Let me construct this carefully. The Lander-Bailey algorithm uses:
    # - 4 "diagonal" terms
    # - 15 other terms with specific patterns
    
    # Actually, let me use a verified rank-19 decomposition.
    # I'll use the one from the paper "Rank of 4x4x4 matrix multiplication" 
    # The coefficients are known but complex. Let me use a different approach.
    
    # Alternative: Use Strassen's rank-7 decomposition for 2x2x2, then tensor it.
    # Strassen 2x2x2: rank 7
    # For 4x4x4, we can view it as 2x2 blocks of 2x2 matrices.
    # The block-level decomposition has rank 7 (Strassen on 2x2 blocks).
    # Each block-level term involves 2x2x2 multiplication, which has rank 7.
    # So total rank = 7 * 7 = 49.
    
    # This gives rank 49 for 4x4x4, which is better than schoolbook 64.
    # Then add 16 schoolbook terms for the 5th column: total rank 65.
    
    # Let me implement Strassen-based rank-49 for 4x4x4.
    
    # Strassen 2x2x2 decomposition:
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B12-B22)
    # M4 = A22(B21-B11)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    # C11 = M1+M4-M5+M7
    # C12 = M3+M5
    # C21 = M2+M4
    # C22 = M1-M2+M3+M6
    
    # U_2: (4, 7), V_2: (4, 7), W_2: (4, 7)
    # Row index: i*2+j for U, j*2+k for V, k*2+i for W
    
    U_2 = [
        [1,  1,  0,  0,  0,  0,  0],  # A11: M1
        [0,  0,  1,  0,  0,  0,  0],  # A12: M3
        [0,  0,  0,  0,  0,  1,  0],  # A21: M2
        [1,  1,  0,  0,  0,  0,  0],  # A22: M1
    ]
    
    # Wait, let me be more careful.
    # A = [[A11, A12], [A21, A22]]
    # B = [[B11, B12], [B21, B22]]
    # U rows: A11, A12, A21, A22 (indices 0,1,2,3)
    # V rows: B11, B12, B21, B22 (indices 0,1,2,3)
    # W rows: C11, C12, C21, C22 (indices 0,1,2,3)
    
    # M1 = (A11+A22)(B11+B22)
    #   U: A11=1, A22=1
    #   V: B11=1, B22=1
    #   W: C11=1, C22=1
    # M2 = (A21+A22)B11
    #   U: A21=1, A22=1
    #   V: B11=1
    #   W: C21=1
    # M3 = A11(B12-B22)
    #   U: A11=1
    #   V: B12=1, B22=-1
    #   W: C12=1
    # M4 = A22(B21-B11)
    #   U: A22=1
    #   V: B21=1, B11=-1
    #   W: C11=1, C21=1
    # M5 = (A11+A12)B22
    #   U: A11=1, A12=1
    #   V: B22=1
    #   W: C11=-1, C12=1
    # M6 = (A21-A11)(B11+B12)
    #   U: A21=1, A11=-1
    #   V: B11=1, B12=1
    #   W: C22=1
    # M7 = (A12-A22)(B21+B22)
    #   U: A12=1, A22=-1
    #   V: B21=1, B22=1
    #   W: C11=1, C22=1
    
    U_2 = np.zeros((4, 7), dtype=int)
    V_2 = np.zeros((4, 7), dtype=int)
    W_2 = np.zeros((4, 7), dtype=int)
    
    # M1
    U_2[0, 0] = 1; U_2[3, 0] = 1
    V_2[0, 0] = 1; V_2[3, 0] = 1
    W_2[0, 0] = 1; W_2[3, 0] = 1
    
    # M2
    U_2[2, 1] = 1; U_2[3, 1] = 1
    V_2[0, 1] = 1
    W_2[2, 1] = 1
    
    # M3
    U_2[0, 2] = 1
    V_2[1, 2] = 1; V_2[3, 2] = -1
    W_2[1, 2] = 1
    
    # M4
    U_2[3, 3] = 1
    V_2[2, 3] = 1; V_2[0, 3] = -1
    W_2[0, 3] = 1; W_2[2, 3] = 1
    
    # M5
    U_2[0, 4] = 1; U_2[1, 4] = 1
    V_2[3, 4] = 1
    W_2[0, 4] = -1; W_2[1, 4] = 1
    
    # M6
    U_2[2, 5] = 1; U_2[0, 5] = -1
    V_2[0, 5] = 1; V_2[1, 5] = 1
    W_2[3, 5] = 1
    
    # M7
    U_2[1, 6] = 1; U_2[3, 6] = -1
    V_2[2, 6] = 1; V_2[3, 6] = 1
    W_2[0, 6] = 1; W_2[3, 6] = 1
    
    # Now for 4x4x4 using block Strassen:
    # A (4x4) as 2x2 blocks of 2x2:
    # A = [[A11, A12], [A21, A22]] where each block is 2x2
    # B (4x4) as 2x2 blocks of 2x2:
    # B = [[B11, B12], [B21, B22]] where each block is 2x2
    # C (4x4) as 2x2 blocks of 2x2:
    # C = [[C11, C12], [C21, C22]] where each block is 2x2
    
    # Block-level: C_block = A_block * B_block using Strassen (rank 7)
    # Each block multiplication is 2x2x2 (rank 7)
    # Total rank: 7 * 7 = 49
    
    # U_4: (16, 49)
    # V_4: (16, 49)
    # W_4: (16, 49)
    
    # U_4 row index: i*4+j where i,j in 0..3
    # Block indices: I=i//2, J=j//2, i2=i%2, j2=j%2
    # So U_4[i*4+j, l*7+s] = U_block[l, I*2+J] * U_2[i2*2+j2, s]
    
    # Similarly for V and W.
    
    U_4 = np.zeros((16, 49), dtype=int)
    V_4 = np.zeros((16, 49), dtype=int)
    W_4 = np.zeros((16, 49), dtype=int)
    
    for l in range(7):
        for s in range(7):
            for I in range(2):
                for J in range(2):
                    for i2 in range(2):
                        for j2 in range(2):
                            i = I * 2 + i2
                            j = J * 2 + j2
                            u_idx = i * 4 + j
                            u_block_idx = I * 2 + J
                            u_2_idx = i2 * 2 + j2
                            U_4[u_idx, l * 7 + s] = U_2[u_2_idx, s] * U_2[u_block_idx, l]
    
    for l in range(7):
        for s in range(7):
            for J in range(2):
                for K in range(2):
                    for j2 in range(2):
                        for k2 in range(2):
                            j = J * 2 + j2
                            k = K * 2 + k2
                            v_idx = j * 4 + k
                            v_block_idx = J * 2 + K
                            v_2_idx = j2 * 2 + k2
                            V_4[v_idx, l * 7 + s] = V_2[v_2_idx, s] * V_2[v_block_idx, l]
    
    for l in range(7):
        for s in range(7):
            for K in range(2):
                for I in range(2):
                    for k2 in range(2):
                        for i2 in range(2):
                            k = K * 2 + k2
                            i = I * 2 + i2
                            w_idx = k * 4 + i
                            w_block_idx = K * 2 + I
                            w_2_idx = k2 * 2 + i2
                            W_4[w_idx, l * 7 + s] = W_2[w_2_idx, s] * W_2[w_block_idx, l]
    
    # Now extend to 4x4x5:
    # A is 4x4, B is 4x5, C is 4x5
    # We handle the first 4 columns of B using the rank-49 decomposition above.
    # For the 5th column of B (index 4), we use schoolbook: 16 terms.
    
    # Total rank: 49 + 16 = 65
    
    rank = 65
    U = np.zeros((n * m, rank), dtype=int)
    V = np.zeros((m * p, rank), dtype=int)
    W = np.zeros((p * n, rank), dtype=int)
    
    # First 49 columns: 4x4x4 part
    U[:, :49] = U_4
    V[:16, :49] = V_4  # V is (20, 65), first 16 rows for 4x4 part
    W[:16, :49] = W_4  # W is (20, 65), first 16 rows for 4x4 part
    
    # Last 16 columns: schoolbook for 4x4x1 (5th column of B)
    for i in range(4):
        for j in range(4):
            col = 49 + i * 4 + j
            U[i * 4 + j, col] = 1
            V[j * 5 + 4, col] = 1  # B's 5th column (index 4)
            W[4 * 4 + i, col] = 1  # C's 5th column: k=4, so index 4*4+i
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
