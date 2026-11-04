import numpy as np

DIMENSIONS = [3, 5, 7]

def _schoolbook_factors(n, m, p):
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    return U, V, W

def _strassen_factors():
    # Factors for 2x2x2 Strassen
    # M = A B
    # A = [[A11, A12], [A21, A22]]
    # B = [[B11, B12], [B21, B22]]
    # C = [[C11, C12], [C21, C22]]
    # 7 products:
    # P1 = (A11 + A22)(B11 + B22)
    # P2 = (A21 + A22) B11
    # P3 = A11 (B12 - B22)
    # P4 = A22 (B21 - B11)
    # P5 = (A11 + A12) B22
    # P6 = (A21 - A11) (B11 + B12)
    # P7 = (A12 - A22) (B21 + B22)
    # C11 = P1 + P4 - P5 + P7
    # C12 = P3 + P5
    # C21 = P2 + P4
    # C22 = P1 - P2 + P3 + P6
    
    # U maps A entries to P inputs
    # V maps B entries to P inputs
    # W maps P outputs to C entries
    
    # A indices: 0:(1,1), 1:(1,2), 2:(2,1), 3:(2,2)
    # B indices: 0:(1,1), 1:(1,2), 2:(2,1), 3:(2,2)
    # C indices: 0:(1,1), 1:(1,2), 2:(2,1), 3:(2,2)
    
    U = np.zeros((4, 7))
    V = np.zeros((4, 7))
    W = np.zeros((4, 7))
    
    # P1
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[3, 0] = 1
    W[0, 0] = 1; W[3, 0] = 1
    
    # P2
    U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1
    W[2, 1] = 1; W[3, 1] = -1 # C21 gets P2, C22 gets -P2
    
    # P3
    U[0, 2] = 1
    V[1, 2] = 1; V[3, 2] = -1
    W[0, 2] = -1; W[1, 2] = 1 # C11 gets -P3? No.
    # C11 = P1 + P4 - P5 + P7. C12 = P3 + P5.
    # So W[0, 2] should be 0?
    # Let's re-verify C11.
    # C11 = P1 + P4 - P5 + P7.
    # C12 = P3 + P5.
    # C21 = P2 + P4.
    # C22 = P1 - P2 + P3 + P6.
    
    # Let's re-fill W carefully.
    W = np.zeros((4, 7))
    # Col 0 (P1): C11+=1, C22+=1
    W[0, 0] = 1; W[3, 0] = 1
    # Col 1 (P2): C21+=1, C22+=-1
    W[2, 1] = 1; W[3, 1] = -1
    # Col 2 (P3): C12+=1, C22+=1
    W[1, 2] = 1; W[3, 2] = 1
    # Col 3 (P4): C11+=1, C21+=1
    W[0, 3] = 1; W[2, 3] = 1
    # Col 4 (P5): C11+=-1, C12+=1
    W[0, 4] = -1; W[1, 4] = 1
    # Col 5 (P6): C22+=1
    W[3, 5] = 1
    # Col 6 (P7): C11+=1
    W[0, 6] = 1
    
    # U and V
    U = np.zeros((4, 7))
    V = np.zeros((4, 7))
    # P1: (A11+A22)(B11+B22)
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[3, 0] = 1
    # P2: (A21+A22)B11
    U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1
    # P3: A11(B12-B22)
    U[0, 2] = 1
    V[1, 2] = 1; V[3, 2] = -1
    # P4: A22(B21-B11)
    U[3, 3] = 1
    V[2, 3] = 1; V[0, 3] = -1
    # P5: (A11+A12)B22
    U[0, 4] = 1; U[1, 4] = 1
    V[3, 4] = 1
    # P6: (A21-A11)(B11+B12)
    U[2, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[1, 5] = 1
    # P7: (A12-A22)(B21+B22)
    U[1, 6] = 1; U[3, 6] = -1
    V[2, 6] = 1; V[3, 6] = 1
    
    return U, V, W

def _embed(U_sub, V_sub, W_sub, i_off, j_off, k_off, n, m, p, n_sub, m_sub, p_sub):
    """
    Embeds sub-factors into global factors.
    """
    U_global = np.zeros((n * m, U_sub.shape[1]))
    V_global = np.zeros((m * p, V_sub.shape[1]))
    W_global = np.zeros((p * n, W_sub.shape[1]))
    
    for col in range(U_sub.shape[1]):
        for i_local in range(n_sub):
            for j_local in range(m_sub):
                val = U_sub[i_local * m_sub + j_local, col]
                if val != 0:
                    i_global = i_off + i_local
                    j_global = j_off + j_local
                    U_global[i_global * m + j_global, col] = val
                    
        for j_local in range(m_sub):
            for k_local in range(p_sub):
                val = V_sub[j_local * p_sub + k_local, col]
                if val != 0:
                    j_global = j_off + j_local
                    k_global = k_off + k_local
                    V_global[j_global * p + k_global, col] = val
                    
        for k_local in range(p_sub):
            for i_local in range(n_sub):
                val = W_sub[k_local * n_sub + i_local, col]
                if val != 0:
                    k_global = k_off + k_local
                    i_global = i_off + i_local
                    W_global[k_global * n + i_global, col] = val
                    
    return U_global, V_global, W_global

def construct():
    n, m, p = DIMENSIONS
    
    # Decomposition:
    # M_3,5,7 = M_2,5,7 (rows 0,1) + M_1,5,7 (row 2)
    # M_2,5,7 = M_2,2,7 (cols 0,1) + M_2,2,7 (cols 2,3) + M_2,1,7 (col 4)
    # M_2,2,7 = 3 * M_2,2,2 (Strassen) + M_2,2,1 (Schoolbook)
    # M_2,2,1 = Schoolbook
    # M_2,1,7 = Schoolbook
    # M_1,5,7 = Schoolbook
    
    blocks = []
    
    # Add M_2,5,7 blocks
    # 1. M_2,2,7 at i=0, j=0, k=0
    #    Split k=0..6 into 2,2,2,1
    #    k=0,1 -> Strassen (2,2,2)
    #    k=2,3 -> Strassen (2,2,2)
    #    k=4,5 -> Strassen (2,2,2)
    #    k=6   -> Schoolbook (2,2,1)
    
    # We will create a list of (U, V, W, i_off, j_off, k_off, n_sub, m_sub, p_sub, method)
    
    # Block 1: i=0, j=0, k=0, size 2,2,2, Strassen
    blocks.append((0, 0, 0, 2, 2, 2, 'strassen'))
    # Block 2: i=0, j=0, k=2, size 2,2,2, Strassen
    blocks.append((0, 0, 2, 2, 2, 2, 'strassen'))
    # Block 3: i=0, j=0, k=4, size 2,2,2, Strassen
    blocks.append((0, 0, 4, 2, 2, 2, 'strassen'))
    # Block 4: i=0, j=0, k=6, size 2,2,1, Schoolbook
    blocks.append((0, 0, 6, 2, 2, 1, 'schoolbook'))
    
    # Block 5: i=0, j=2, k=0, size 2,2,2, Strassen
    blocks.append((0, 2, 0, 2, 2, 2, 'strassen'))
    # Block 6: i=0, j=2, k=2, size 2,2,2, Strassen
    blocks.append((0, 2, 2, 2, 2, 2, 'strassen'))
    # Block 7: i=0, j=2, k=4, size 2,2,2, Strassen
    blocks.append((0, 2, 4, 2, 2, 2, 'strassen'))
    # Block 8: i=0, j=2, k=6, size 2,2,1, Schoolbook
    blocks.append((0, 2, 6, 2, 2, 1, 'schoolbook'))
    
    # Block 9: i=0, j=4, k=0, size 2,1,7, Schoolbook
    blocks.append((0, 4, 0, 2, 1, 7, 'schoolbook'))
    
    # Block 10: i=2, j=0, k=0, size 1,5,7, Schoolbook
    blocks.append((2, 0, 0, 1, 5, 7, 'schoolbook'))
    
    total_rank = 0
    for _, _, _, n_s, m_s, p_s, method in blocks:
        if method == 'strassen':
            total_rank += 7
        else:
            total_rank += n_s * m_s * p_s
            
    U = np.zeros((n * m, total_rank))
    V = np.zeros((m * p, total_rank))
    W = np.zeros((p * n, total_rank))
    
    col_offset = 0
    for i_off, j_off, k_off, n_s, m_s, p_s, method in blocks:
        if method == 'strassen':
            U_s, V_s, W_s = _strassen_factors()
        else:
            U_s, V_s, W_s = _schoolbook_factors(n_s, m_s, p_s)
            
        U_e, V_e, W_e = _embed(U_s, V_s, W_s, i_off, j_off, k_off, n, m, p, n_s, m_s, p_s)
        
        U[:, col_offset:col_offset+U_s.shape[1]] = U_e
        V[:, col_offset:col_offset+V_s.shape[1]] = V_e
        W[:, col_offset:col_offset+W_s.shape[1]] = W_e
        
        col_offset += U_s.shape[1]
        
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
