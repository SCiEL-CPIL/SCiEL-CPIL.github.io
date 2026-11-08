import numpy as np

DIMENSIONS = [2, 4, 4]

def construct():
    n, m, p = DIMENSIONS
    # Rank 28 solution using 4x Strassen (2x2x2)
    # Strassen rank is 7 for 2x2x2
    
    # Define the 7 rank-1 tensors for 2x2x2 multiplication
    # U, V, W are 2x2 matrices for each term
    # Indices: i (0,1), j (0,1), k (0,1)
    
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B01-B11)
    # M4 = A11(B10-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    U_2x2 = []
    V_2x2 = []
    W_2x2 = []
    
    # Term 1
    U1 = np.array([[1, 0], [0, 1]])
    V1 = np.array([[1, 0], [0, 1]])
    W1 = np.array([[1, 0], [0, 1]]) # C00=1, C11=1
    U_2x2.append(U1); V_2x2.append(V1); W_2x2.append(W1)
    
    # Term 2
    U2 = np.array([[0, 0], [1, 1]])
    V2 = np.array([[1, 0], [0, 0]])
    W2 = np.array([[0, 0], [1, 0]]) # C10=1
    U_2x2.append(U2); V_2x2.append(V2); W_2x2.append(W2)
    
    # Term 3
    U3 = np.array([[1, 0], [0, 0]])
    V3 = np.array([[0, 1], [0, -1]])
    W3 = np.array([[0, 1], [0, 0]]) # C01=1
    U_2x2.append(U3); V_2x2.append(V3); W_2x2.append(W3)
    
    # Term 4
    U4 = np.array([[0, 0], [0, 1]])
    V4 = np.array([[0, 0], [1, -1]])
    W4 = np.array([[1, 0], [1, 0]]) # C00=1, C10=1
    U_2x2.append(U4); V_2x2.append(V4); W_2x2.append(W4)
    
    # Term 5
    U5 = np.array([[1, 1], [0, 0]])
    V5 = np.array([[0, 0], [0, 1]])
    W5 = np.array([[-1, 1], [0, 0]]) # C00=-1, C01=1
    U_2x2.append(U5); V_2x2.append(V5); W_2x2.append(W5)
    
    # Term 6
    U6 = np.array([[-1, 0], [1, 0]])
    V6 = np.array([[1, 1], [0, 0]])
    W6 = np.array([[0, 0], [0, 1]]) # C11=1
    U_2x2.append(U6); V_2x2.append(V6); W_2x2.append(W6)
    
    # Term 7
    U7 = np.array([[0, 1], [0, -1]])
    V7 = np.array([[0, 0], [1, 1]])
    W7 = np.array([[1, 0], [0, 0]]) # C00=1
    U_2x2.append(U7); V_2x2.append(V7); W_2x2.append(W7)

    # Now build the full 2x4x4 decomposition
    # 4 blocks:
    # 1: A[0:2, 0:2] * B[0:2, 0:2] -> C[0:2, 0:2]
    # 2: A[0:2, 2:4] * B[2:4, 0:2] -> C[0:2, 0:2]
    # 3: A[0:2, 0:2] * B[0:2, 2:4] -> C[0:2, 2:4]
    # 4: A[0:2, 2:4] * B[2:4, 2:4] -> C[0:2, 2:4]
    
    total_rank = 28
    U = np.zeros((n*m, total_rank))
    V = np.zeros((m*p, total_rank))
    W = np.zeros((p*n, total_rank))
    
    # Block 1: i_off=0, j_off=0, k_off=0
    for r in range(7):
        for i in range(2):
            for j in range(2):
                U[i*4 + (0+j), r] = U_2x2[r][i, j]
        for j in range(2):
            for k in range(2):
                V[(0+j)*4 + (0+k), r] = V_2x2[r][j, k]
        for k in range(2):
            for i in range(2):
                W[(0+k)*2 + (0+i), r] = W_2x2[r][k, i]
                
    # Block 2: i_off=0, j_off=2, k_off=0
    for r in range(7):
        col_offset = 7 + r
        for i in range(2):
            for j in range(2):
                U[i*4 + (2+j), col_offset] = U_2x2[r][i, j]
        for j in range(2):
            for k in range(2):
                V[(2+j)*4 + (0+k), col_offset] = V_2x2[r][j, k]
        for k in range(2):
            for i in range(2):
                W[(0+k)*2 + (0+i), col_offset] = W_2x2[r][k, i]

    # Block 3: i_off=0, j_off=0, k_off=2
    for r in range(7):
        col_offset = 14 + r
        for i in range(2):
            for j in range(2):
                U[i*4 + (0+j), col_offset] = U_2x2[r][i, j]
        for j in range(2):
            for k in range(2):
                V[(0+j)*4 + (2+k), col_offset] = V_2x2[r][j, k]
        for k in range(2):
            for i in range(2):
                W[(2+k)*2 + (0+i), col_offset] = W_2x2[r][k, i]

    # Block 4: i_off=0, j_off=2, k_off=2
    for r in range(7):
        col_offset = 21 + r
        for i in range(2):
            for j in range(2):
                U[i*4 + (2+j), col_offset] = U_2x2[r][i, j]
        for j in range(2):
            for k in range(2):
                V[(2+j)*4 + (2+k), col_offset] = V_2x2[r][j, k]
        for k in range(2):
            for i in range(2):
                W[(2+k)*2 + (0+i), col_offset] = W_2x2[r][k, i]

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
