import numpy as np

DIMENSIONS = [2, 4, 7]

def _get_strassen_222():
    """Returns U, V, W for 2x2x2 multiplication using Strassen's algorithm (Rank 7)."""
    n, m, p = 2, 2, 2
    r = 7
    U = np.zeros((n * m, r))
    V = np.zeros((m * p, r))
    W = np.zeros((p * n, r))
    
    # M1 = (A00+A11)(B00+B11)
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[3, 0] = 1
    W[0, 0] = 1; W[3, 0] = 1
    
    # M2 = (A10+A11)B00
    U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1
    W[1, 1] = 1; W[3, 1] = -1
    
    # M3 = A00(B01-B11)
    U[0, 2] = 1
    V[1, 2] = 1; V[3, 2] = -1
    W[2, 2] = 1; W[3, 2] = 1
    
    # M4 = A11(B10-B00)
    U[3, 3] = 1
    V[2, 3] = 1; V[0, 3] = -1
    W[0, 3] = 1; W[1, 3] = 1
    
    # M5 = (A00+A01)B11
    U[0, 4] = 1; U[1, 4] = 1
    V[3, 4] = 1
    W[0, 4] = -1; W[2, 4] = 1
    
    # M6 = (A10-A00)(B00+B01)
    U[2, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[1, 5] = 1
    W[3, 5] = 1
    
    # M7 = (A01-A11)(B10+B11)
    U[1, 6] = 1; U[3, 6] = -1
    V[2, 6] = 1; V[3, 6] = 1
    W[0, 6] = 1
    
    return U, V, W

def _get_ones_221():
    """Returns U, V, W for 2x2x1 multiplication using rank-1 decomposition."""
    n, m, p = 2, 2, 1
    r = 1
    U = np.ones((n * m, r))
    V = np.ones((m * p, r))
    W = np.ones((p * n, r))
    return U, V, W

def construct():
    n, m, p = DIMENSIONS
    
    # Split m=4 into m1=2, m2=2.
    # For each m-block, split p=7 into 2+2+2+1.
    # Ranks: 7+7+7+1 = 22 per m-block
    # Total rank = 2 * 22 = 44
    
    total_rank = 44
    U = np.zeros((n * m, total_rank))
    V = np.zeros((m * p, total_rank))
    W = np.zeros((p * n, total_rank))
    
    col_offset = 0
    
    def place_subtensor(U_sub, V_sub, W_sub, i_offset, j_offset, k_offset):
        """Place a sub-tensor at the given offsets."""
        r_sub = U_sub.shape[1]
        n_sub = U_sub.shape[0] // j_offset  # This is getting complicated
        # Better to just do explicit loops
        pass
    
    # Block 1: m in [0, 2)
    # Sub-blocks for p: [0,2), [2,4), [4,6), [6,7)
    
    p_splits = [(0, 2), (2, 4), (4, 6), (6, 7)]
    
    for (k_start, k_end) in p_splits:
        k_len = k_end - k_start
        if k_len == 2:
            U_sub, V_sub, W_sub = _get_strassen_222()
        else:  # k_len == 1
            U_sub, V_sub, W_sub = _get_ones_221()
        
        r_sub = U_sub.shape[1]
        
        # Place U: U indices i*4+j, j in [0,2)
        for i in range(2):
            for j in range(2):
                U[i*4 + j, col_offset:col_offset+r_sub] = U_sub[i*2 + j, :]
        
        # Place V: V indices j*7+k, j in [0,2), k in [k_start, k_end)
        for j in range(2):
            for k in range(k_len):
                V[j*7 + (k_start + k), col_offset:col_offset+r_sub] = V_sub[j*k_len + k, :]
        
        # Place W: W indices k*2+i, k in [k_start, k_end), i in [0,2)
        for k in range(k_len):
            for i in range(2):
                W[(k_start + k)*2 + i, col_offset:col_offset+r_sub] = W_sub[k*2 + i, :]
        
        col_offset += r_sub
    
    # Block 2: m in [2, 4)
    for (k_start, k_end) in p_splits:
        k_len = k_end - k_start
        if k_len == 2:
            U_sub, V_sub, W_sub = _get_strassen_222()
        else:  # k_len == 1
            U_sub, V_sub, W_sub = _get_ones_221()
        
        r_sub = U_sub.shape[1]
        
        # Place U: U indices i*4+j, j in [2,4)
        for i in range(2):
            for j in range(2):
                U[i*4 + (j+2), col_offset:col_offset+r_sub] = U_sub[i*2 + j, :]
        
        # Place V: V indices j*7+k, j in [2,4), k in [k_start, k_end)
        for j in range(2):
            for k in range(k_len):
                V[(j+2)*7 + (k_start + k), col_offset:col_offset+r_sub] = V_sub[j*k_len + k, :]
        
        # Place W: W indices k*2+i, k in [k_start, k_end), i in [0,2)
        for k in range(k_len):
            for i in range(2):
                W[(k_start + k)*2 + i, col_offset:col_offset+r_sub] = W_sub[k*2 + i, :]
        
        col_offset += r_sub
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
