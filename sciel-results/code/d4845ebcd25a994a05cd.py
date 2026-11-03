<<<<<<< SEARCH
def _build_rank50():
    """Build the rank-50 block decomposition."""
    n, m, p = 2, 4, 7
    total_rank = 50
    U = np.zeros((n * m, total_rank))
    V = np.zeros((m * p, total_rank))
    W = np.zeros((p * n, total_rank))
    
    col_offset = 0
    m_blocks = [(0, 2), (2, 4)]
    p_blocks = [
        (0, 2, 'strassen'),
        (2, 4, 'strassen'),
        (4, 6, 'strassen'),
        (6, 7, 'school')
    ]
    
    U_s, V_s, W_s = _get_strassen_222()
    U_sc, V_sc, W_sc = _get_school_221()
    
    for j_start, j_end in m_blocks:
        for k_start, k_end, p_type in p_blocks:
            if p_type == 'strassen':
                U_sub, V_sub, W_sub = U_s, V_s, W_s
                r_sub = 7
            else:
                U_sub, V_sub, W_sub = U_sc, V_sc, W_sc
                r_sub = 4
            
            for i in range(n):
                for j_sub in range(j_end - j_start):
                    j = j_start + j_sub
                    U[i * m + j, col_offset:col_offset + r_sub] = U_sub[i * 2 + j_sub, :]
            
            p_sub = k_end - k_start
            for j_sub in range(j_end - j_start):
                j = j_start + j_sub
                for k_sub in range(p_sub):
                    k = k_start + k_sub
                    V[j * p + k, col_offset:col_offset + r_sub] = V_sub[j_sub * p_sub + k_sub, :]
            
            for k_sub in range(p_sub):
                k = k_start + k_sub
                for i in range(n):
                    W[k * n + i, col_offset:col_offset + r_sub] = W_sub[k_sub * 2 + i, :]
            
            col_offset += r_sub
    
    return U, V, W
=======
def _get_strassen_222():
    """
    Returns U, V, W for 2x2x2 multiplication using Strassen's algorithm (Rank 7).
    U shape: (4, 7), V shape: (4, 7), W shape: (4, 7)
    Indices: U[i*2+j], V[j*2+k], W[k*2+i]
    """
    n, m, p = 2, 2, 2
    r = 7
    U = np.zeros((n * m, r))
    V = np.zeros((m * p, r))
    W = np.zeros((p * n, r))
    
    # Strassen terms M1..M7
    # M1 = (A00+A11)(B00+B11) -> C00 += M1, C11 += M1
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[3, 0] = 1
    W[0, 0] = 1; W[3, 0] = 1
    
    # M2 = (A10+A11)B00 -> C10 += M2, C11 -= M2
    U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1
    W[1, 1] = 1; W[3, 1] = -1
    
    # M3 = A00(B01-B11) -> C01 += M3, C11 += M3
    U[0, 2] = 1
    V[1, 2] = 1; V[3, 2] = -1
    W[2, 2] = 1; W[3, 2] = 1
    
    # M4 = A11(B10-B00) -> C00 += M4, C10 += M4
    U[3, 3] = 1
    V[2, 3] = 1; V[0, 3] = -1
    W[0, 3] = 1; W[1, 3] = 1
    
    # M5 = (A00+A01)B11 -> C00 -= M5, C01 += M5
    U[0, 4] = 1; U[1, 4] = 1
    V[3, 4] = 1
    W[0, 4] = -1; W[2, 4] = 1
    
    # M6 = (A10-A00)(B00+B01) -> C11 += M6
    U[2, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[1, 5] = 1
    W[3, 5] = 1
    
    # M7 = (A01-A11)(B10+B11) -> C00 += M7
    U[1, 6] = 1; U[3, 6] = -1
    V[2, 6] = 1; V[3, 6] = 1
    W[0, 6] = 1
    
    return U, V, W

def _get_school_222():
    """
    Returns U, V, W for 2x2x2 multiplication using schoolbook (Rank 4).
    U shape: (4, 4), V shape: (4, 4), W shape: (4, 4)
    """
    n, m, p = 2, 2, 2
    r = 4
    U = np.zeros((n * m, r))
    V = np.zeros((m * p, r))
    W = np.zeros((p * n, r))
    
    for c in range(r):
        i = c // m
        j = c % m
        k = 0
        U[i * m + j, c] = 1.0
        V[j * p + k, c] = 1.0
        W[k * n + i, c] = 1.0
        
    return U, V, W

def _build_223():
    """
    Build 2x2x3 decomposition: Strassen (rank 7) for cols 0,1 + schoolbook (rank 4) for col 2.
    Total rank 11.
    U shape: (4, 11), V shape: (6, 11), W shape: (6, 11)
    """
    n, m, p = 2, 2, 3
    r = 11
    U = np.zeros((n * m, r))
    V = np.zeros((m * p, r))
    W = np.zeros((p * n, r))
    
    # Part 1: Strassen for columns 0,1 (rank 7)
    U_s, V_s, W_s = _get_strassen_222()
    U[:, 0:7] = U_s
    V[:, 0:7] = V_s
    W[:, 0:7] = W_s
    
    # Part 2: Schoolbook for column 2 (rank 4)
    # C[0,2] = A[0,0]*B[0,2] + A[0,1]*B[1,2]
    # C[1,2] = A[1,0]*B[0,2] + A[1,1]*B[1,2]
    for c in range(4):
        i = c // 2
        j = c % 2
        col = 7 + c
        U[i * 2 + j, col] = 1.0
        V[j * 3 + 2, col] = 1.0  # B[j, 2]
        W[2 * 2 + i, col] = 1.0  # C[i, 2] -> W[k*n+i] with k=2
    
    return U, V, W

def _build_224():
    """
    Build 2x2x4 decomposition: Strassen (rank 7) for cols 0,1 + schoolbook (rank 4) for cols 2,3.
    Total rank 11.
    U shape: (4, 11), V shape: (8, 11), W shape: (8, 11)
    """
    n, m, p = 2, 2, 4
    r = 11
    U = np.zeros((n * m, r))
    V = np.zeros((m * p, r))
    W = np.zeros((p * n, r))
    
    # Part 1: Strassen for columns 0,1 (rank 7)
    U_s, V_s, W_s = _get_strassen_222()
    U[:, 0:7] = U_s
    V[:, 0:7] = V_s
    W[:, 0:7] = W_s
    
    # Part 2: Schoolbook for columns 2,3 (rank 4)
    # For col 2: C[0,2]=A[0,0]*B[0,2]+A[0,1]*B[1,2], C[1,2]=A[1,0]*B[0,2]+A[1,1]*B[1,2]
    # For col 3: C[0,3]=A[0,0]*B[0,3]+A[0,1]*B[1,3], C[1,3]=A[1,0]*B[0,3]+A[1,1]*B[1,3]
    # We can share the A combinations:
    # Term 7: A[0,0] * (B[0,2], B[0,3]) -> contributes to C[0,2] and C[0,3]
    # Term 8: A[0,1] * (B[1,2], B[1,3]) -> contributes to C[0,2] and C[0,3]
    # Term 9: A[1,0] * (B[0,2], B[0,3]) -> contributes to C[1,2] and C[1,3]
    # Term 10: A[1,1] * (B[1,2], B[1,3]) -> contributes to C[1,2] and C[1,3]
    for c in range(4):
        i = c // 2
        j = c % 2
        col = 7 + c
        U[i * 2 + j, col] = 1.0
        # This term handles both col 2 and col 3 for this (i,j) pair
        V[j * 4 + 2, col] = 1.0  # B[j, 2]
        V[j * 4 + 3, col] = 1.0  # B[j, 3]
        W[2 * 2 + i, col] = 1.0  # C[i, 2]
        W[3 * 2 + i, col] = 1.0  # C[i, 3]
    
    return U, V, W

def _build_rank44():
    """Build the rank-44 block decomposition using 2x2x3 and 2x2x4 blocks."""
    n, m, p = 2, 4, 7
    total_rank = 44
    U = np.zeros((n * m, total_rank))
    V = np.zeros((m * p, total_rank))
    W = np.zeros((p * n, total_rank))
    
    col_offset = 0
    m_blocks = [(0, 2), (2, 4)]
    p_blocks = [
        (0, 3, '223'),
        (3, 7, '224')
    ]
    
    U_223, V_223, W_223 = _build_223()
    U_224, V_224, W_224 = _build_224()
    
    for j_start, j_end in m_blocks:
        for k_start, k_end, p_type in p_blocks:
            if p_type == '223':
                U_sub, V_sub, W_sub = U_223, V_223, W_223
                r_sub = 11
                p_sub = 3
            else:
                U_sub, V_sub, W_sub = U_224, V_224, W_224
                r_sub = 11
                p_sub = 4
            
            for i in range(n):
                for j_sub in range(j_end - j_start):
                    j = j_start + j_sub
                    U[i * m + j, col_offset:col_offset + r_sub] = U_sub[i * 2 + j_sub, :]
            
            for j_sub in range(j_end - j_start):
                j = j_start + j_sub
                for k_sub in range(p_sub):
                    k = k_start + k_sub
                    V[j * p + k, col_offset:col_offset + r_sub] = V_sub[j_sub * p_sub + k_sub, :]
            
            for k_sub in range(p_sub):
                k = k_start + k_sub
                for i in range(n):
                    W[k * n + i, col_offset:col_offset + r_sub] = W_sub[k_sub * 2 + i, :]
            
            col_offset += r_sub
    
    return U, V, W
>>>>>>> REPLACE

Now I need to update the `construct` function to use the new rank-44 builder:

