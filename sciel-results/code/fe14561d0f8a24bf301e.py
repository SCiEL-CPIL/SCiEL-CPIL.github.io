import numpy as np
import time

def construct():
    n, m, p = 4, 4, 8
    deadline = time.time() + 140
    
    # Schoolbook fallback
    rank_school = n * m * p
    U_school = np.zeros((n * m, rank_school))
    V_school = np.zeros((m * p, rank_school))
    W_school = np.zeros((p * n, rank_school))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j, col] = 1
                V_school[j * p + k, col] = 1
                W_school[k * n + i, col] = 1
                col += 1
    
    best_U, best_V, best_W = U_school, V_school, W_school
    best_rank = rank_school

    # Define Strassen 2x2 factors
    U2 = np.zeros((4, 7))
    V2 = np.zeros((4, 7))
    W2 = np.zeros((4, 7))
    
    # t=0
    U2[0, 0] = 1; U2[3, 0] = 1
    V2[0, 0] = 1; V2[3, 0] = 1
    W2[0, 0] = 1; W2[3, 0] = 1
    
    # t=1
    U2[2, 1] = 1; U2[3, 1] = 1
    V2[0, 1] = 1
    W2[2, 1] = 1; W2[3, 1] = -1
    
    # t=2
    U2[0, 2] = 1
    V2[1, 2] = 1; V2[3, 2] = -1
    W2[1, 2] = 1; W2[3, 2] = 1
    
    # t=3
    U2[3, 3] = 1
    V2[2, 3] = 1; V2[0, 3] = -1
    W2[0, 3] = 1; W2[2, 3] = 1
    
    # t=4
    U2[0, 4] = 1; U2[1, 4] = 1
    V2[3, 4] = 1
    W2[0, 4] = -1; W2[1, 4] = 1
    
    # t=5
    U2[2, 5] = 1; U2[0, 5] = -1
    V2[0, 5] = 1; V2[1, 5] = 1
    W2[3, 5] = 1
    
    # t=6
    U2[1, 6] = 1; U2[3, 6] = -1
    V2[2, 6] = 1; V2[3, 6] = 1
    W2[0, 6] = 1
    
    # Construct 4x4 factors (Rank 49) using Kronecker product logic
    R44 = 49
    U44 = np.zeros((16, R44))
    V44 = np.zeros((16, R44))
    W44 = np.zeros((16, R44))
    
    # Reshape to 2x2x7 to use broadcasting
    # U2 is 4x7 (2x2 flattened). We want U44[i,j, t1,t2] = U2[i1,j1,t1] * U2[i2,j2,t2]
    # Indices: i = i1*2 + i2, j = j1*2 + j2.
    # U2 index is i1*2 + j1.
    
    # Create index grids
    i1 = np.array([0,0,1,1]) # for i=0,1,2,3 -> i1=0,0,1,1
    i2 = np.array([0,1,0,1])
    j1 = np.array([0,1,0,1]) # for j=0,1,2,3 -> j1=0,1,0,1
    j2 = np.array([0,0,1,1])
    
    # This mapping is tricky.
    # i=0 -> i1=0, i2=0
    # i=1 -> i1=0, i2=1
    # i=2 -> i1=1, i2=0
    # i=3 -> i1=1, i2=1
    # So i1 = i // 2, i2 = i % 2.
    
    i_idx = np.arange(4)
    i1_idx = i_idx // 2
    i2_idx = i_idx % 2
    j1_idx = i_idx // 2
    j2_idx = i_idx % 2
    
    # U2[i1*2+j1, t]
    # We need to construct U44[i,j, t1*7+t2]
    
    # Vectorized construction
    for t1 in range(7):
        for t2 in range(7):
            t44 = t1 * 7 + t2
            # U2 values for t1, t2
            u1 = U2[:, t1] # shape (4,)
            u2 = U2[:, t2] # shape (4,)
            v1 = V2[:, t1]
            v2 = V2[:, t2]
            w1 = W2[:, t1]
            w2 = W2[:, t2]
            
            # Construct U44[:, t44]
            # U44[i,j] = U2[i1*2+j1, t1] * U2[i2*2+j2, t2]
            # i1*2+j1 is the index in U2 for the first block.
            # We can reshape U2 to 2x2x7.
            U2_reshaped = U2.reshape(2, 2, 7)
            V2_reshaped = V2.reshape(2, 2, 7)
            W2_reshaped = W2.reshape(2, 2, 7)
            
            # u1[i1, j1] = U2_reshaped[i1, j1, t1]
            # We want U44[i,j] where i=i1*2+i2, j=j1*2+j2
            # This is effectively a Kronecker product of the 2x2 matrices U2[:,:,t1] and U2[:,:,t2]
            
            U_mat1 = U2_reshaped[:,:,t1]
            U_mat2 = U2_reshaped[:,:,t2]
            U44_mat = np.kron(U_mat1, U_mat2)
            U44[:, t44] = U44_mat.flatten()
            
            V_mat1 = V2_reshaped[:,:,t1]
            V_mat2 = V2_reshaped[:,:,t2]
            V44_mat = np.kron(V_mat1, V_mat2)
            V44[:, t44] = V44_mat.flatten()
            
            W_mat1 = W2_reshaped[:,:,t1]
            W_mat2 = W2_reshaped[:,:,t2]
            W44_mat = np.kron(W_mat1, W_mat2)
            W44[:, t44] = W44_mat.flatten()
            
    # Verify 4x4 decomposition
    valid_44 = True
    for i in range(4):
        for j in range(4):
            for k in range(4):
                val = np.dot(U44[i * 4 + j, :], V44[j * 4 + k, :] * W44[k * 4 + i, :])
                if abs(val - 1.0) > 1e-9:
                    valid_44 = False
                    break
            if not valid_44: break
        if not valid_44: break
        
    if valid_44:
        R448 = 98
        U448 = np.zeros((16, R448))
        V448 = np.zeros((32, R448))
        W448 = np.zeros((32, R448))
        
        U448[:, :49] = U44
        W448[:, :49] = W44
        for j in range(4):
            for k in range(4):
                V448[j * 8 + k, :49] = V44[j * 4 + k, :49]
                
        U448[:, 49:] = U44
        W448[:, 49:] = W44
        for j in range(4):
            for k in range(4):
                V448[j * 8 + (k + 4), 49:] = V44[j * 4 + k, :49]
                
        valid_448 = True
        for i in range(4):
            for j in range(4):
                for k in range(8):
                    val = np.dot(U448[i * 4 + j, :], V448[j * 8 + k, :] * W448[k * 4 + i, :])
                    if abs(val - 1.0) > 1e-9:
                        valid_448 = False
                        break
                if not valid_448: break
            if not valid_448: break
            
        if valid_448:
            best_U, best_V, best_W = U448, V448, W448
            best_rank = R448

    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
