import numpy as np

def construct():
    n, m, p = 5, 5, 5
    
    # 2x2 Strassen Decomposition
    # U2, V2, W2 are 4x7
    U2 = np.zeros((4, 7), dtype=float)
    V2 = np.zeros((4, 7), dtype=float)
    W2 = np.zeros((4, 7), dtype=float)
    
    # M1: (A11+A22)(B11+B22)
    U2[0, 0] = 1; U2[3, 0] = 1
    V2[0, 0] = 1; V2[3, 0] = 1
    W2[0, 0] = 1; W2[1, 0] = 1; W2[2, 0] = 1; W2[3, 0] = 1
    
    # M2: (A21+A22)B11
    U2[2, 1] = 1; U2[3, 1] = 1
    V2[0, 1] = 1
    W2[2, 1] = 1; W2[3, 1] = 1
    
    # M3: A11(B12-B22)
    U2[0, 2] = 1
    V2[1, 2] = 1; V2[3, 2] = -1
    W2[1, 2] = 1; W2[3, 2] = 1
    
    # M4: A22(B21-B11)
    U2[3, 3] = 1
    V2[0, 3] = -1; V2[2, 3] = 1
    W2[0, 3] = 1; W2[2, 3] = 1
    
    # M5: (A11+A12)B22
    U2[0, 4] = 1; U2[1, 4] = 1
    V2[3, 4] = 1
    W2[0, 4] = 1; W2[1, 4] = -1
    
    # M6: (A21-A11)(B11+B12)
    U2[0, 5] = -1; U2[2, 5] = 1
    V2[0, 5] = 1; V2[1, 5] = 1
    W2[3, 5] = 1
    
    # M7: (A12-A22)(B21+B22)
    U2[1, 6] = 1; U2[3, 6] = -1
    V2[2, 6] = 1; V2[3, 6] = 1
    W2[0, 6] = 1; W2[3, 6] = 1

    # 4x4 Strassen Decomposition (Rank 49)
    # U44, V44, W44 are 16x49
    U44 = np.zeros((16, 49), dtype=float)
    V44 = np.zeros((16, 49), dtype=float)
    W44 = np.zeros((16, 49), dtype=float)
    
    for c1 in range(7):
        for c2 in range(7):
            col = c1 * 7 + c2
            for I in range(2):
                for i in range(2):
                    global_i = I * 2 + i
                    # U44[global_i, col] = U2[global_i, c1] * U2[i, c2]
                    U44[global_i, col] = U2[global_i, c1] * U2[i, c2]
                    V44[global_i, col] = V2[global_i, c1] * V2[i, c2]
                    W44[global_i, col] = W2[global_i, c1] * W2[i, c2]
                    
    # Total Rank = 49 + 61 = 110
    total_rank = 110
    U = np.zeros((25, total_rank), dtype=float)
    V = np.zeros((25, total_rank), dtype=float)
    W = np.zeros((25, total_rank), dtype=float)
    
    # Copy 4x4 block (indices 0-3)
    # U[0:16, 0:49] = U44
    U[:16, :49] = U44
    V[:16, :49] = V44
    W[:16, :49] = W44
    
    col_idx = 49
    
    # Border Terms
    # 1. A[i,j] B[j,4] for i,j in 0..3
    for i in range(4):
        for j in range(4):
            U[i*5+j, col_idx] = 1
            V[j*5+4, col_idx] = 1
            W[4*5+i, col_idx] = 1
            col_idx += 1
            
    # 2. A[i,4] B[4,j] for i,j in 0..3
    for i in range(4):
        for j in range(4):
            U[i*5+4, col_idx] = 1
            V[4*5+j, col_idx] = 1
            W[j*5+i, col_idx] = 1
            col_idx += 1
            
    # 3. A[4,j] B[j,k] for j,k in 0..3
    for j in range(4):
        for k in range(4):
            U[4*5+j, col_idx] = 1
            V[j*5+k, col_idx] = 1
            W[k*5+4, col_idx] = 1
            col_idx += 1
            
    # 4. A[4,i] B[i,4] for i in 0..3
    for i in range(4):
        U[4*5+i, col_idx] = 1
        V[i*5+4, col_idx] = 1
        W[4*5+4, col_idx] = 1
        col_idx += 1
        
    # 5. A[i,4] B[4,4] for i in 0..3
    for i in range(4):
        U[i*5+4, col_idx] = 1
        V[4*5+4, col_idx] = 1
        W[4*5+i, col_idx] = 1
        col_idx += 1
        
    # 6. A[4,4] B[4,j] for j in 0..3
    for j in range(4):
        U[4*5+4, col_idx] = 1
        V[4*5+j, col_idx] = 1
        W[j*5+4, col_idx] = 1
        col_idx += 1
        
    # 7. A[4,4] B[4,4]
    U[4*5+4, col_idx] = 1
    V[4*5+4, col_idx] = 1
    W[4*5+4, col_idx] = 1
    col_idx += 1
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
