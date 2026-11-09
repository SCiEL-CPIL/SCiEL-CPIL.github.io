"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 3, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    
    # Helper to create a zero matrix
    def zero_matrix(rows, cols):
        return [[0] * cols for _ in range(rows)]
    
    # Helper to add a block of matrices into global matrices at a specific offset
    def add_block(global_U, global_V, global_W, local_U, local_V, local_W, 
                  i_offset, j_offset, k_offset, rank_offset, 
                  local_n, local_m, local_p):
        local_rank = len(local_U[0])
        for r in range(local_rank):
            for i in range(local_n):
                for j in range(local_m):
                    local_u_idx = i * local_m + j
                    global_u_idx = (i + i_offset) * m + (j + j_offset)
                    global_U[global_u_idx][rank_offset + r] += local_U[local_u_idx][r]
                    
            for j in range(local_m):
                for k in range(local_p):
                    local_v_idx = j * local_p + k
                    global_v_idx = (j + j_offset) * p + (k + k_offset)
                    global_V[global_v_idx][rank_offset + r] += local_V[local_v_idx][r]
                    
            for k in range(local_p):
                for i in range(local_n):
                    local_w_idx = k * local_n + i
                    global_w_idx = (k + k_offset) * n + (i + i_offset)
                    global_W[global_w_idx][rank_offset + r] += local_W[local_w_idx][r]
                    
        return local_rank

    # Strassen 2x2x2 decomposition (Rank 7)
    def strassen_2x2x2():
        n_l, m_l, p_l = 2, 2, 2
        rank = 7
        U = zero_matrix(n_l * m_l, rank)
        V = zero_matrix(m_l * p_l, rank)
        W = zero_matrix(p_l * n_l, rank)
        
        # M1 = (A00 + A11)(B00 + B11)
        U[0][0] = 1; U[3][0] = 1
        V[0][0] = 1; V[3][0] = 1
        W[0][0] = 1; W[3][0] = 1; W[1][0] = -1; W[2][0] = -1
        
        # M2 = (A10 + A11)B00
        U[2][1] = 1; U[3][1] = 1
        V[0][1] = 1
        W[2][1] = 1; W[3][1] = 1
        
        # M3 = A00(B10 - B11)
        U[0][2] = 1
        V[2][2] = 1; V[3][2] = -1
        W[0][2] = 1; W[1][2] = 1
        
        # M4 = A11(B01 - B00)
        U[3][3] = 1
        V[1][3] = 1; V[0][3] = -1
        W[1][3] = 1; W[3][3] = 1
        
        # M5 = (A00 + A01)B11
        U[0][4] = 1; U[1][4] = 1
        V[3][4] = 1
        W[3][4] = 1; W[2][4] = -1
        
        # M6 = (A10 - A00)(B00 + B01)
        U[2][5] = 1; U[0][5] = -1
        V[0][5] = 1; V[1][5] = 1
        W[2][5] = 1; W[0][5] = -1
        
        # M7 = (A01 - A11)(B10 + B11)
        U[1][6] = 1; U[3][6] = -1
        V[2][6] = 1; V[3][6] = 1
        W[1][6] = 1; W[3][6] = -1
        
        return U, V, W

    # Schoolbook 2x2x1 decomposition (Rank 4)
    def schoolbook_2x2x1():
        n_l, m_l, p_l = 2, 2, 1
        rank = 4
        U = zero_matrix(n_l * m_l, rank)
        V = zero_matrix(m_l * p_l, rank)
        W = zero_matrix(p_l * n_l, rank)
        
        col = 0
        for i in range(n_l):
            for j in range(m_l):
                for k in range(p_l):
                    U[i * m_l + j][col] = 1
                    V[j * p_l + k][col] = 1
                    W[k * n_l + i][col] = 1
                    col += 1
        return U, V, W

    # Rank 1 decomposition for 2x1x7
    def rank1_2x1x7():
        n_l, m_l, p_l = 2, 1, 7
        rank = 1
        U = zero_matrix(n_l * m_l, rank)
        V = zero_matrix(m_l * p_l, rank)
        W = zero_matrix(p_l * n_l, rank)
        
        for i in range(n_l):
            U[i * m_l][0] = 1
        for k in range(p_l):
            V[k][0] = 1
            W[k * n_l][0] = 1 # i=0
            W[k * n_l + 1][0] = 1 # i=1
        return U, V, W

    # Rank 2 decomposition for 1x2x7
    def rank2_1x2x7():
        n_l, m_l, p_l = 1, 2, 7
        rank = 2
        U = zero_matrix(n_l * m_l, rank)
        V = zero_matrix(m_l * p_l, rank)
        W = zero_matrix(p_l * n_l, rank)
        
        # Term 1: j=0
        U[0][0] = 1
        for k in range(p_l):
            V[k][0] = 1
            W[k][0] = 1
        # Term 2: j=1
        U[1][1] = 1
        for k in range(p_l):
            V[p_l + k][1] = 1
            W[k][1] = 1
        return U, V, W

    # Rank 1 decomposition for 1x1x7
    def rank1_1x1x7():
        n_l, m_l, p_l = 1, 1, 7
        rank = 1
        U = zero_matrix(n_l * m_l, rank)
        V = zero_matrix(m_l * p_l, rank)
        W = zero_matrix(p_l * n_l, rank)
        
        U[0][0] = 1
        for k in range(p_l):
            V[k][0] = 1
            W[k][0] = 1
        return U, V, W

    # Initialize global matrices
    total_rank = 0
    # Pre-calculate total rank to allocate memory
    r_222 = 7
    r_221 = 4
    r_217 = 1
    r_127 = 2
    r_117 = 1
    total_rank = 3 * r_222 + r_221 + r_217 + r_127 + r_117
    
    U = zero_matrix(n * m, total_rank)
    V = zero_matrix(m * p, total_rank)
    W = zero_matrix(p * n, total_rank)
    
    offset = 0
    
    # Block 1: i in {0,1}, j in {0,1} -> 2x2x7
    # Sub-block 1.1: 2x2x2 (k in {0,1})
    U11, V11, W11 = strassen_2x2x2()
    offset += add_block(U, V, W, U11, V11, W11, 0, 0, 0, offset, 2, 2, 2)
    
    # Sub-block 1.2: 2x2x2 (k in {2,3})
    U12, V12, W12 = strassen_2x2x2()
    offset += add_block(U, V, W, U12, V12, W12, 0, 0, 2, offset, 2, 2, 2)
    
    # Sub-block 1.3: 2x2x2 (k in {4,5})
    U13, V13, W13 = strassen_2x2x2()
    offset += add_block(U, V, W, U13, V13, W13, 0, 0, 4, offset, 2, 2, 2)
    
    # Sub-block 1.4: 2x2x1 (k in {6})
    U14, V14, W14 = schoolbook_2x2x1()
    offset += add_block(U, V, W, U14, V14, W14, 0, 0, 6, offset, 2, 2, 1)
    
    # Block 2: i in {0,1}, j in {2} -> 2x1x7
    U2, V2, W2 = rank1_2x1x7()
    offset += add_block(U, V, W, U2, V2, W2, 0, 2, 0, offset, 2, 1, 7)
    
    # Block 3: i in {2}, j in {0,1} -> 1x2x7
    U3, V3, W3 = rank2_1x2x7()
    offset += add_block(U, V, W, U3, V3, W3, 2, 0, 0, offset, 1, 2, 7)
    
    # Block 4: i in {2}, j in {2} -> 1x1x7
    U4, V4, W4 = rank1_1x1x7()
    offset += add_block(U, V, W, U4, V4, W4, 2, 2, 0, offset, 1, 1, 7)
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
