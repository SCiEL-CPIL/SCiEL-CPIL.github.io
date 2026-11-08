import numpy as np
import time

DIMENSIONS = [3, 4, 5]
_START_TIME = time.time()
_DEADLINE = _START_TIME + 100  # 50s safety margin before 150s limit

def construct():
    """
    Constructs an exact tensor decomposition for matrix multiplication of size 3x4x5.
    Rank 56.
    """
    n, m, p = DIMENSIONS
    
    if time.time() > _DEADLINE:
        # Fallback to schoolbook rank 60 if we run out of time (shouldn't happen)
        return _schoolbook_fallback(n, m, p)

    # Pre-allocate matrices
    # U: (n*m, rank), V: (m*p, rank), W: (p*n, rank)
    # We know the rank is 56.
    rank = 56
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    current_rank = 0
    
    def add_block(u_idx, v_idx, w_idx, U_blk, V_blk, W_blk):
        nonlocal current_rank
        for r in range(U_blk.shape[1]):
            for i, idx in enumerate(u_idx):
                U[idx, current_rank] = U_blk[i, r]
            for i, idx in enumerate(v_idx):
                V[idx, current_rank] = V_blk[i, r]
            for i, idx in enumerate(w_idx):
                W[idx, current_rank] = W_blk[i, r]
            current_rank += 1

    # 1. Strassen 2x2x2 decomposition
    def strassen_2x2x2():
        U = np.zeros((4, 7))
        V = np.zeros((4, 7))
        W = np.zeros((4, 7))
        
        # M1 = (A00+A11)(B00+B11)
        U[0, 0] = 1; U[3, 0] = 1
        V[0, 0] = 1; V[3, 0] = 1
        W[0, 0] = 1; W[3, 0] = 1
        
        # M2 = (A10+A11)B00
        U[2, 1] = 1; U[3, 1] = 1
        V[0, 1] = 1
        W[2, 1] = 1; W[3, 1] = -1
        
        # M3 = A00(B01-B11)
        U[0, 2] = 1
        V[1, 2] = 1; V[3, 2] = -1
        W[1, 2] = 1; W[3, 2] = 1
        
        # M4 = A11(B10-B00)
        U[3, 3] = 1
        V[2, 3] = 1; V[0, 3] = -1
        W[0, 3] = 1; W[2, 3] = 1
        
        # M5 = (A00+A01)B11
        U[0, 4] = 1; U[1, 4] = 1
        V[3, 4] = 1
        W[0, 4] = -1; W[1, 4] = 1
        
        # M6 = (A10-A00)(B00+B01)
        U[2, 5] = 1; U[0, 5] = -1
        V[0, 5] = 1; V[1, 5] = 1
        W[3, 5] = 1
        
        # M7 = (A01-A11)(B10+B11)
        U[1, 6] = 1; U[3, 6] = -1
        V[2, 6] = 1; V[3, 6] = 1
        W[0, 6] = 1
        
        return U, V, W
    
    # 2. Schoolbook 2x2x1 decomposition
    def schoolbook_2x2x1():
        U = np.zeros((4, 4))
        V = np.zeros((2, 4))
        W = np.zeros((2, 4))
        
        U[0, 0] = 1; V[0, 0] = 1; W[0, 0] = 1
        U[1, 1] = 1; V[1, 1] = 1; W[0, 1] = 1
        U[2, 2] = 1; V[0, 2] = 1; W[1, 2] = 1
        U[3, 3] = 1; V[1, 3] = 1; W[1, 3] = 1
        
        return U, V, W
    
    # 3. Schoolbook 1x2x2 decomposition
    def schoolbook_1x2x2():
        U = np.zeros((2, 4))
        V = np.zeros((4, 4))
        W = np.zeros((2, 4))
        
        U[0, 0] = 1; V[0, 0] = 1; W[0, 0] = 1
        U[1, 1] = 1; V[2, 1] = 1; W[0, 1] = 1
        U[0, 2] = 1; V[1, 2] = 1; W[1, 2] = 1
        U[1, 3] = 1; V[3, 3] = 1; W[1, 3] = 1
        
        return U, V, W
    
    # 4. Schoolbook 1x2x1 decomposition
    def schoolbook_1x2x1():
        U = np.zeros((2, 2))
        V = np.zeros((2, 2))
        W = np.zeros((1, 2))
        
        U[0, 0] = 1; V[0, 0] = 1; W[0, 0] = 1
        U[1, 1] = 1; V[1, 1] = 1; W[0, 1] = 1
        
        return U, V, W

    # Indices for U (n*m = 12), V (m*p = 20), W (p*n = 15)
    # U[i*m + j]
    # V[i*m + j]
    # W[k*n + i]
    
    # Part 1: Rows 0,1 of A (2x4)
    # Split into two 2x2 blocks along columns of A and rows of B.
    # Block 1.1: A[0:2, 0:2] * B[0:2, :]
    # A[0:2, 0:2] indices: 0, 1, 4, 5
    # B[0:2, :] indices: 0,1,2,3,4, 5,6,7,8,9
    # Split B[0:2, :] into B[0:2, 0:2], B[0:2, 2:4], B[0:2, 4:5]
    
    # 1.1.1: A[0:2, 0:2] * B[0:2, 0:2] (2x2x2)
    u_idx = [0, 1, 4, 5]
    v_idx = [0, 1, 5, 6]
    w_idx = [0, 3, 1, 4] # C[0,0], C[0,1], C[1,0], C[1,1]
    U_blk, V_blk, W_blk = strassen_2x2x2()
    add_block(u_idx, v_idx, w_idx, U_blk, V_blk, W_blk)
    
    # 1.1.2: A[0:2, 0:2] * B[0:2, 2:4] (2x2x2)
    u_idx = [0, 1, 4, 5]
    v_idx = [2, 3, 7, 8]
    w_idx = [6, 9, 7, 10] # C[0,2], C[0,3], C[1,2], C[1,3]
    U_blk, V_blk, W_blk = strassen_2x2x2()
    add_block(u_idx, v_idx, w_idx, U_blk, V_blk, W_blk)
    
    # 1.1.3: A[0:2, 0:2] * B[0:2, 4:5] (2x2x1)
    u_idx = [0, 1, 4, 5]
    v_idx = [4, 9]
    w_idx = [12, 13] # C[0,4], C[1,4]
    U_blk, V_blk, W_blk = schoolbook_2x2x1()
    add_block(u_idx, v_idx, w_idx, U_blk, V_blk, W_blk)
    
    # Block 1.2: A[0:2, 2:4] * B[2:4, :]
    # A[0:2, 2:4] indices: 2, 3, 6, 7
    # B[2:4, :] indices: 10,11,12,13,14, 15,16,17,18,19
    # Split B[2:4, :] into B[2:4, 0:2], B[2:4, 2:4], B[2:4, 4:5]
    
    # 1.2.1: A[0:2, 2:4] * B[2:4, 0:2] (2x2x2)
    u_idx = [2, 3, 6, 7]
    v_idx = [10, 11, 15, 16]
    w_idx = [0, 3, 1, 4]
    U_blk, V_blk, W_blk = strassen_2x2x2()
    add_block(u_idx, v_idx, w_idx, U_blk, V_blk, W_blk)
    
    # 1.2.2: A[0:2, 2:4] * B[2:4, 2:4] (2x2x2)
    u_idx = [2, 3, 6, 7]
    v_idx = [12, 13, 17, 18]
    w_idx = [6, 9, 7, 10]
    U_blk, V_blk, W_blk = strassen_2x2x2()
    add_block(u_idx, v_idx, w_idx, U_blk, V_blk, W_blk)
    
    # 1.2.3: A[0:2, 2:4] * B[2:4, 4:5] (2x2x1)
    u_idx = [2, 3, 6, 7]
    v_idx = [14, 19]
    w_idx = [12, 13]
    U_blk, V_blk, W_blk = schoolbook_2x2x1()
    add_block(u_idx, v_idx, w_idx, U_blk, V_blk, W_blk)
    
    # Part 2: Row 2 of A (1x4)
    # Split into two 1x2 blocks along columns of A and rows of B.
    # Block 2.1: A[2, 0:2] * B[0:2, :]
    # A[2, 0:2] indices: 8, 9
    # B[0:2, :] indices: 0,1,2,3,4, 5,6,7,8,9
    
    # 2.1.1: A[2, 0:2] * B[0:2, 0:2] (1x2x2)
    u_idx = [8, 9]
    v_idx = [0, 1, 5, 6]
    w_idx = [2, 5] # C[2,0], C[2,1]
    U_blk, V_blk, W_blk = schoolbook_1x2x2()
    add_block(u_idx, v_idx, w_idx, U_blk, V_blk, W_blk)
    
    # 2.1.2: A[2, 0:2] * B[0:2, 2:4] (1x2x2)
    u_idx = [8, 9]
    v_idx = [2, 3, 7, 8]
    w_idx = [8, 11] # C[2,2], C[2,3]
    U_blk, V_blk, W_blk = schoolbook_1x2x2()
    add_block(u_idx, v_idx, w_idx, U_blk, V_blk, W_blk)
    
    # 2.1.3: A[2, 0:2] * B[0:2, 4:5] (1x2x1)
    u_idx = [8, 9]
    v_idx = [4, 9]
    w_idx = [14] # C[2,4]
    U_blk, V_blk, W_blk = schoolbook_1x2x1()
    add_block(u_idx, v_idx, w_idx, U_blk, V_blk, W_blk)
    
    # Block 2.2: A[2, 2:4] * B[2:4, :]
    # A[2, 2:4] indices: 10, 11
    # B[2:4, :] indices: 10,11,12,13,14, 15,16,17,18,19
    
    # 2.2.1: A[2, 2:4] * B[2:4, 0:2] (1x2x2)
    u_idx = [10, 11]
    v_idx = [10, 11, 15, 16]
    w_idx = [2, 5]
    U_blk, V_blk, W_blk = schoolbook_1x2x2()
    add_block(u_idx, v_idx, w_idx, U_blk, V_blk, W_blk)
    
    # 2.2.2: A[2, 2:4] * B[2:4, 2:4] (1x2x2)
    u_idx = [10, 11]
    v_idx = [12, 13, 17, 18]
    w_idx = [8, 11]
    U_blk, V_blk, W_blk = schoolbook_1x2x2()
    add_block(u_idx, v_idx, w_idx, U_blk, V_blk, W_blk)
    
    # 2.2.3: A[2, 2:4] * B[2:4, 4:5] (1x2x1)
    u_idx = [10, 11]
    v_idx = [14, 19]
    w_idx = [14]
    U_blk, V_blk, W_blk = schoolbook_1x2x1()
    add_block(u_idx, v_idx, w_idx, U_blk, V_blk, W_blk)
    
    # Final check
    if current_rank != 56:
        # This should not happen
        pass
        
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }

def _schoolbook_fallback(n, m, p):
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    idx = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, idx] = 1
                V[j * p + k, idx] = 1
                W[k * n + i, idx] = 1
                idx += 1
                
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
