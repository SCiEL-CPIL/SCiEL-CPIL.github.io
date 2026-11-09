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
    # U, V, W are 4x7 matrices (flattened 2x2)
    # Indexing: U[i*2+j, t], V[j*2+k, t], W[k*2+i, t]
    
    U2 = np.zeros((4, 7))
    V2 = np.zeros((4, 7))
    W2 = np.zeros((4, 7))
    
    # t=0
    U2[0, 0] = 1; U2[3, 0] = 1
    V2[0, 0] = 1; V2[3, 0] = 1
    W2[0, 0] = 1; W2[3, 0] = 1 # W[k*2+i]: C00 (k=0,i=0)->0, C11 (k=1,i=1)->3
    
    # t=1
    U2[2, 1] = 1; U2[3, 1] = 1
    V2[0, 1] = 1
    W2[2, 1] = 1; W2[3, 1] = -1 # C10 (k=0,i=1)->2, C11 (k=1,i=1)->3
    
    # t=2
    U2[0, 2] = 1
    V2[1, 2] = 1; V2[3, 2] = -1
    W2[1, 2] = 1; W2[3, 2] = 1 # C01 (k=1,i=0)->1, C11 (k=1,i=1)->3
    
    # t=3
    U2[3, 3] = 1
    V2[2, 3] = 1; V2[0, 3] = -1
    W2[0, 3] = 1; W2[2, 3] = 1 # C00 (k=0,i=0)->0, C10 (k=0,i=1)->2
    
    # t=4
    U2[0, 4] = 1; U2[1, 4] = 1
    V2[3, 4] = 1
    W2[0, 4] = -1; W2[1, 4] = 1 # C00 (k=0,i=0)->0, C01 (k=1,i=0)->1
    
    # t=5
    U2[2, 5] = 1; U2[0, 5] = -1
    V2[0, 5] = 1; V2[1, 5] = 1
    W2[3, 5] = 1 # C11 (k=1,i=1)->3
    
    # t=6
    U2[1, 6] = 1; U2[3, 6] = -1
    V2[2, 6] = 1; V2[3, 6] = 1
    W2[0, 6] = 1 # C00 (k=0,i=0)->0
    
    # Construct 4x4 factors (Rank 49)
    R44 = 49
    U44 = np.zeros((16, R44))
    V44 = np.zeros((16, R44))
    W44 = np.zeros((16, R44))
    
    for t1 in range(7):
        for t2 in range(7):
            t44 = t1 * 7 + t2
            for i in range(4):
                I = i // 2
                i_loc = i % 2
                for j in range(4):
                    J = j // 2
                    j_loc = j % 2
                    U44[i * 4 + j, t44] = U2[I * 2 + J, t1] * U2[i_loc * 2 + j_loc, t2]
                    
            for j in range(4):
                J = j // 2
                j_loc = j % 2
                for k in range(4):
                    K = k // 2
                    k_loc = k % 2
                    V44[j * 4 + k, t44] = V2[J * 2 + K, t1] * V2[j_loc * 2 + k_loc, t2]
                    
            for k in range(4):
                K = k // 2
                k_loc = k % 2
                for i in range(4):
                    I = i // 2
                    i_loc = i % 2
                    W44[k * 4 + i, t44] = W2[K * 2 + I, t1] * W2[k_loc * 2 + i_loc, t2]
                    
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
        # Construct 4x4x8 factors (Rank 98)
        R448 = 98
        U448 = np.zeros((16, R448))
        V448 = np.zeros((32, R448))
        W448 = np.zeros((32, R448))
        
        # First half (B1)
        U448[:, :49] = U44
        W448[:, :49] = W44
        for j in range(4):
            for k in range(4):
                V448[j * 8 + k, :49] = V44[j * 4 + k, :49]
                
        # Second half (B2)
        U448[:, 49:] = U44
        W448[:, 49:] = W44
        for j in range(4):
            for k in range(4):
                V448[j * 8 + (k + 4), 49:] = V44[j * 4 + k, :49]
                
        # Verify 4x4x8
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
