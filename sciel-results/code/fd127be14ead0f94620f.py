import time
import numpy as np

DIMENSIONS = [2, 4, 7]

def construct():
    """
    Returns a rank-50 exact tensor decomposition for 2x4 x 4x7 matrix multiplication.
    Uses Strassen (rank 7) for 2x2x2 blocks and Schoolbook (rank 4) for 2x2x1 blocks.
    """
    n, m, p = 2, 4, 7
    total_rank = 50
    
    U = np.zeros((n * m, total_rank))  # 8 x 50
    V = np.zeros((m * p, total_rank))  # 28 x 50
    W = np.zeros((p * n, total_rank))  # 14 x 50
    
    def get_strassen_222():
        """Returns U, V, W for 2x2x2 multiplication using Strassen's algorithm (Rank 7)."""
        r = 7
        U_sub = np.zeros((4, r))
        V_sub = np.zeros((4, r))
        W_sub = np.zeros((4, r))
        
        # Indices: U[i*2+j], V[j*2+k], W[k*2+i]
        # Strassen terms M1..M7
        
        # M1 = (A00+A11)(B00+B11)
        U_sub[0, 0] = 1; U_sub[3, 0] = 1
        V_sub[0, 0] = 1; V_sub[3, 0] = 1
        W_sub[0, 0] = 1; W_sub[3, 0] = 1
        
        # M2 = (A10+A11)B00
        U_sub[2, 1] = 1; U_sub[3, 1] = 1
        V_sub[0, 1] = 1
        W_sub[1, 1] = 1; W_sub[3, 1] = -1
        
        # M3 = A00(B01-B11)
        U_sub[0, 2] = 1
        V_sub[1, 2] = 1; V_sub[3, 2] = -1
        W_sub[2, 2] = 1; W_sub[3, 2] = 1
        
        # M4 = A11(B10-B00)
        U_sub[3, 3] = 1
        V_sub[2, 3] = 1; V_sub[0, 3] = -1
        W_sub[0, 3] = 1; W_sub[1, 3] = 1
        
        # M5 = (A00+A01)B11
        U_sub[0, 4] = 1; U_sub[1, 4] = 1
        V_sub[3, 4] = 1
        W_sub[0, 4] = -1; W_sub[2, 4] = 1
        
        # M6 = (A10-A00)(B00+B01)
        U_sub[2, 5] = 1; U_sub[0, 5] = -1
        V_sub[0, 5] = 1; V_sub[1, 5] = 1
        W_sub[3, 5] = 1
        
        # M7 = (A01-A11)(B10+B11)
        U_sub[1, 6] = 1; U_sub[3, 6] = -1
        V_sub[2, 6] = 1; V_sub[3, 6] = 1
        W_sub[0, 6] = 1
        
        return U_sub, V_sub, W_sub
    
    def get_school_221():
        """Returns U, V, W for 2x2x1 multiplication using schoolbook (Rank 4)."""
        r = 4
        U_sub = np.zeros((4, r))
        V_sub = np.zeros((2, r))
        W_sub = np.zeros((2, r))
        
        for c in range(r):
            i = c // 2
            j = c % 2
            U_sub[i * 2 + j, c] = 1.0
            V_sub[j, c] = 1.0
            W_sub[i, c] = 1.0
            
        return U_sub, V_sub, W_sub
    
    def place_block(U_sub, V_sub, W_sub, m_start, p_start, col_offset):
        """
        Places a sub-decomposition into the global arrays.
        m_start: starting index in the middle dimension (0 or 2)
        p_start: starting index in the output dimension (0, 2, 4, or 6)
        """
        r_sub = U_sub.shape[1]
        
        # Place U: U[i*m + j, col] = U_sub[i*2 + j', col]
        # j = j' + m_start
        for i in range(2):
            for j_prime in range(2):
                j = j_prime + m_start
                U[i * m + j, col_offset:col_offset + r_sub] = U_sub[i * 2 + j_prime, :]
        
        # Place V: V[j*p + k, col] = V_sub[j'*2 + k', col]
        # j = j' + m_start, k = k' + p_start
        for j_prime in range(2):
            j = j_prime + m_start
            for k_prime in range(V_sub.shape[0] // 2):
                k = k_prime + p_start
                V[j * p + k, col_offset:col_offset + r_sub] = V_sub[j_prime * 2 + k_prime, :]
        
        # Place W: W[k*n + i, col] = W_sub[k'*2 + i', col]
        # k = k' + p_start
        for k_prime in range(W_sub.shape[0] // 2):
            k = k_prime + p_start
            for i in range(2):
                W[k * n + i, col_offset:col_offset + r_sub] = W_sub[k_prime * 2 + i, :]
        
        return col_offset + r_sub
    
    col = 0
    
    # Block 1: m in [0, 2)
    # Sub-block 1.1: p in [0, 4) -> 2x2x4
    # Split p in [0, 4) into [0, 2) and [2, 4)
    
    # 1.1.1: p in [0, 2) -> 2x2x2 (Strassen)
    U_sub, V_sub, W_sub = get_strassen_222()
    col = place_block(U_sub, V_sub, W_sub, m_start=0, p_start=0, col_offset=col)
    
    # 1.1.2: p in [2, 4) -> 2x2x2 (Strassen)
    U_sub, V_sub, W_sub = get_strassen_222()
    col = place_block(U_sub, V_sub, W_sub, m_start=0, p_start=2, col_offset=col)
    
    # Sub-block 1.2: p in [4, 7) -> 2x2x3
    # Split p in [4, 7) into [4, 6) and [6, 7)
    
    # 1.2.1: p in [4, 6) -> 2x2x2 (Strassen)
    U_sub, V_sub, W_sub = get_strassen_222()
    col = place_block(U_sub, V_sub, W_sub, m_start=0, p_start=4, col_offset=col)
    
    # 1.2.2: p in [6, 7) -> 2x2x1 (Schoolbook)
    U_sub, V_sub, W_sub = get_school_221()
    col = place_block(U_sub, V_sub, W_sub, m_start=0, p_start=6, col_offset=col)
    
    # Block 2: m in [2, 4)
    # Repeat the same structure for the second half of A and B
    
    # 2.1.1: p in [0, 2)
    U_sub, V_sub, W_sub = get_strassen_222()
    col = place_block(U_sub, V_sub, W_sub, m_start=2, p_start=0, col_offset=col)
    
    # 2.1.2: p in [2, 4)
    U_sub, V_sub, W_sub = get_strassen_222()
    col = place_block(U_sub, V_sub, W_sub, m_start=2, p_start=2, col_offset=col)
    
    # 2.2.1: p in [4, 6)
    U_sub, V_sub, W_sub = get_strassen_222()
    col = place_block(U_sub, V_sub, W_sub, m_start=2, p_start=4, col_offset=col)
    
    # 2.2.2: p in [6, 7)
    U_sub, V_sub, W_sub = get_school_221()
    col = place_block(U_sub, V_sub, W_sub, m_start=2, p_start=6, col_offset=col)
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
