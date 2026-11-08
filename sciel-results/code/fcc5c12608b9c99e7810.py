import numpy as np
import time

_START_TIME = time.time()
_TIME_LIMIT = 145.0  # 5s safety margin before 150s kill

# EVOLVE-BLOCK-START
def construct():
    """
    Constructs an exact tensor decomposition for matrix multiplication of size 3x4x5.
    Uses Strassen's algorithm for 2x2x2 blocks and optimized 1x2x2 decompositions
    to reduce the rank.
    The decomposition is given by U, V, W such that T = sum_r U[:, r] outer V[:, r] outer W[:, r].
    Indices:
    U: (n*m, rank)
    V: (m*p, rank)
    W: (p*n, rank) with W[k*n+i, r]
    """
    global _START_TIME
    if time.time() - _START_TIME > _TIME_LIMIT:
        # Return a fallback valid decomposition (schoolbook) if time is running out
        n, m, p = 3, 4, 5
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
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    n, m, p = 3, 4, 5
    
    # Strassen 2x2x2 decomposition
    # U: 4x7, V: 4x7, W: 4x7
    # U indices: 0:A00, 1:A01, 2:A10, 3:A11
    # V indices: 0:B00, 1:B01, 2:B10, 3:B11
    # W indices: 0:C00, 1:C01, 2:C10, 3:C11
    def get_strassen_2x2x2():
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
    
    # Schoolbook 2x2x1 decomposition
    def get_schoolbook_2x2x1():
        U = np.zeros((4, 4))
        V = np.zeros((2, 4))
        W = np.zeros((2, 4))
        
        U[0, 0] = 1; V[0, 0] = 1; W[0, 0] = 1
        U[1, 1] = 1; V[1, 1] = 1; W[0, 1] = 1
        U[2, 2] = 1; V[0, 2] = 1; W[1, 2] = 1
        U[3, 3] = 1; V[1, 3] = 1; W[1, 3] = 1
        
        return U, V, W
    
    # Schoolbook 1x2x2 decomposition
    def get_schoolbook_1x2x2():
        U = np.zeros((2, 4))
        V = np.zeros((4, 4))
        W = np.zeros((2, 4))
        
        U[0, 0] = 1; V[0, 0] = 1; W[0, 0] = 1
        U[1, 1] = 1; V[2, 1] = 1; W[0, 1] = 1
        U[0, 2] = 1; V[1, 2] = 1; W[1, 2] = 1
        U[1, 3] = 1; V[3, 3] = 1; W[1, 3] = 1
        
        return U, V, W
    
    # Schoolbook 1x2x1 decomposition
    def get_schoolbook_1x2x1():
        U = np.zeros((2, 2))
        V = np.zeros((2, 2))
        W = np.zeros((1, 2))
        
        U[0, 0] = 1; V[0, 0] = 1; W[0, 0] = 1
        U[1, 1] = 1; V[1, 1] = 1; W[0, 1] = 1
        
        return U, V, W

    # Now we assemble the 3x4x5 multiplication.
    U_parts = []
    V_parts = []
    W_parts = []
    
    def add_block(u_idx, v_idx, w_idx, U_block, V_block, W_block):
        rank = U_block.shape[1]
        for r in range(rank):
            U_parts.append(np.zeros(n * m))
            for i, idx in enumerate(u_idx):
                U_parts[-1][idx] = U_block[i, r]
            
            V_parts.append(np.zeros(m * p))
            for i, idx in enumerate(v_idx):
                V_parts[-1][idx] = V_block[i, r]
            
            W_parts.append(np.zeros(p * n))
            for i, idx in enumerate(w_idx):
                W_parts[-1][idx] = W_block[i, r]
    
    # Block 1: A1_top * B_top_left (2x2x2)
    U_blk, V_blk, W_blk = get_strassen_2x2x2()
    u_idx = [0, 1, 4, 5]
    v_idx = [0, 1, 5, 6]
    w_idx = [0, 3, 1, 4]
    add_block(u_idx, v_idx, w_idx, U_blk, V_blk, W_blk)
    
    # Block 2: A1_top * B_top_mid (2x2x2)
    U_blk, V_blk, W_blk = get_strassen_2x2x2()
    u_idx = [0, 1, 4, 5]
    v_idx = [2, 3, 7, 8]
    w_idx = [6, 9, 7, 10]
    add_block(u_idx, v_idx, w_idx, U_blk, V_blk, W_blk)
    
    # Block 3: A1_top * B_top_right (2x2x1)
    U_blk, V_blk, W_blk = get_schoolbook_2x2x1()
    u_idx = [0, 1, 4, 5]
    v_idx = [4, 9]
    w_idx = [12, 13]
    add_block(u_idx, v_idx, w_idx, U_blk, V_blk, W_blk)
    
    # Block 4: A1_bottom * B_bottom_left (2x2x2)
    U_blk, V_blk, W_blk = get_strassen_2x2x2()
    u_idx = [2, 3, 6, 7]
    v_idx = [10, 11, 15, 16]
    w_idx = [0, 3, 1, 4]
    add_block(u_idx, v_idx, w_idx, U_blk, V_blk, W_blk)
    
    # Block 5: A1_bottom * B_bottom_mid (2x2x2)
    U_blk, V_blk, W_blk = get_strassen_2x2x2()
    u_idx = [2, 3, 6, 7]
    v_idx = [12, 13, 17, 18]
    w_idx = [6, 9, 7, 10]
    add_block(u_idx, v_idx, w_idx, U_blk, V_blk, W_blk)
    
    # Block 6: A1_bottom * B_bottom_right (2x2x1)
    U_blk, V_blk, W_blk = get_schoolbook_2x2x1()
    u_idx = [2, 3, 6, 7]
    v_idx = [14, 19]
    w_idx = [12, 13]
    add_block(u_idx, v_idx, w_idx, U_blk, V_blk, W_blk)
    
    # Block 7: A2_top * B_top_left (1x2x2)
    U_blk, V_blk, W_blk = get_schoolbook_1x2x2()
    u_idx = [8, 9]
    v_idx = [0, 1, 5, 6]
    w_idx = [2, 5]
    add_block(u_idx, v_idx, w_idx, U_blk, V_blk, W_blk)
    
    # Block 8: A2_top * B_top_mid (1x2x2)
    U_blk, V_blk, W_blk = get_schoolbook_1x2x2()
    u_idx = [8, 9]
    v_idx = [2, 3, 7, 8]
    w_idx = [8, 11]
    add_block(u_idx, v_idx, w_idx, U_blk, V_blk, W_blk)
    
    # Block 9: A2_top * B_top_right (1x2x1)
    U_blk, V_blk, W_blk = get_schoolbook_1x2x1()
    u_idx = [8, 9]
    v_idx = [4, 9]
    w_idx = [14]
    add_block(u_idx, v_idx, w_idx, U_blk, V_blk, W_blk)
    
    # Block 10: A2_bottom * B_bottom_left (1x2x2)
    U_blk, V_blk, W_blk = get_schoolbook_1x2x2()
    u_idx = [10, 11]
    v_idx = [10, 11, 15, 16]
    w_idx = [2, 5]
    add_block(u_idx, v_idx, w_idx, U_blk, V_blk, W_blk)
    
    # Block 11: A2_bottom * B_bottom_mid (1x2x2)
    U_blk, V_blk, W_blk = get_schoolbook_1x2x2()
    u_idx = [10, 11]
    v_idx = [12, 13, 17, 18]
    w_idx = [8, 11]
    add_block(u_idx, v_idx, w_idx, U_blk, V_blk, W_blk)
    
    # Block 12: A2_bottom * B_bottom_right (1x2x1)
    U_blk, V_blk, W_blk = get_schoolbook_1x2x1()
    u_idx = [10, 11]
    v_idx = [14, 19]
    w_idx = [14]
    add_block(u_idx, v_idx, w_idx, U_blk, V_blk, W_blk)
    
    U = np.array(U_parts).T
    V = np.array(V_parts).T
    W = np.array(W_parts).T
    
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
# EVOLVE-BLOCK-END
