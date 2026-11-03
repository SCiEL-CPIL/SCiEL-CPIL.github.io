"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    
    # Helper to create trivial decomposition
    def make_trivial(n_local, m_local, p_local):
        rank = n_local * m_local * p_local
        U = [[0] * rank for _ in range(n_local * m_local)]
        V = [[0] * rank for _ in range(m_local * p_local)]
        W = [[0] * rank for _ in range(p_local * n_local)]
        
        for i in range(n_local):
            for j in range(m_local):
                for k in range(p_local):
                    col = i * m_local * p_local + j * p_local + k
                    u_idx = i * m_local + j
                    v_idx = j * p_local + k
                    w_idx = k * n_local + i
                    U[u_idx][col] = 1
                    V[v_idx][col] = 1
                    W[w_idx][col] = 1
        return U, V, W

    # Helper to create Strassen decomposition for 2x2x2
    def make_strassen():
        # Local dims 2, 2, 2
        n_l, m_l, p_l = 2, 2, 2
        rank = 7
        U = [[0] * rank for _ in range(n_l * m_l)]
        V = [[0] * rank for _ in range(m_l * p_l)]
        W = [[0] * rank for _ in range(p_l * n_l)]
        
        # Indices:
        # U: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
        # V: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
        # W: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)  -- Note: W index is k*n + i
        # k=0, i=0 -> 0
        # k=0, i=1 -> 1
        # k=1, i=0 -> 2
        # k=1, i=1 -> 3
        
        # M1 = (A00+A11)(B00+B11)
        U[0][0] = 1; U[3][0] = 1
        V[0][0] = 1; V[3][0] = 1
        W[0][0] = 1; W[3][0] = 1
        
        # M2 = (A10+A11)B00
        U[2][1] = 1; U[3][1] = 1
        V[0][1] = 1
        W[2][1] = 1; W[3][1] = -1
        
        # M3 = A00(B01-B11)
        U[0][2] = 1
        V[1][2] = 1; V[3][2] = -1
        W[1][2] = 1; W[3][2] = 1
        
        # M4 = A11(B10-B00)
        U[3][3] = 1
        V[2][3] = 1; V[0][3] = -1
        W[0][3] = 1; W[2][3] = 1
        
        # M5 = (A00+A10)B11
        U[0][4] = 1; U[2][4] = 1
        V[3][4] = 1
        W[0][4] = -1; W[1][4] = 1
        
        # M6 = (A01-A00)(B00+B10)
        U[1][5] = 1; U[0][5] = -1
        V[0][5] = 1; V[2][5] = 1
        W[1][5] = 1
        
        # M7 = (A11-A10)(B01+B11)
        U[3][6] = 1; U[2][6] = -1
        V[1][6] = 1; V[3][6] = 1
        W[0][6] = 1; W[3][6] = 1
        
        return U, V, W

    # Combine decompositions
    # sub_decomps: list of (n_l, m_l, p_l, i_off, j_off, k_off, U, V, W)
    sub_decomps = [
        # 1. M_{2,2,2} at (0,0,0)
        (2, 2, 2, 0, 0, 0, *make_strassen()),
        # 2. M_{2,2,2} at (0,0,2)
        (2, 2, 2, 0, 0, 2, *make_strassen()),
        # 3. M_{2,2,3} at (0,0,4) - Trivial
        (2, 2, 3, 0, 0, 4, *make_trivial(2, 2, 3)),
        # 4. M_{2,1,7} at (0,2,0) - Trivial
        (2, 1, 7, 0, 2, 0, *make_trivial(2, 1, 7))
    ]
    
    total_rank = sum(len(u[0]) for _, _, _, _, _, _, u, v, w in sub_decomps)
    
    U_global = [[0] * total_rank for _ in range(n * m)]
    V_global = [[0] * total_rank for _ in range(m * p)]
    W_global = [[0] * total_rank for _ in range(p * n)]
    
    col_offset = 0
    for n_l, m_l, p_l, i_off, j_off, k_off, U_l, V_l, W_l in sub_decomps:
        rank_l = len(U_l[0])
        
        for i_l in range(n_l):
            for j_l in range(m_l):
                u_l_idx = i_l * m_l + j_l
                u_g_idx = (i_off + i_l) * m + (j_off + j_l)
                for c in range(rank_l):
                    U_global[u_g_idx][col_offset + c] = U_l[u_l_idx][c]
        
        for j_l in range(m_l):
            for k_l in range(p_l):
                v_l_idx = j_l * p_l + k_l
                v_g_idx = (j_off + j_l) * p + (k_off + k_l)
                for c in range(rank_l):
                    V_global[v_g_idx][col_offset + c] = V_l[v_l_idx][c]
        
        for k_l in range(p_l):
            for i_l in range(n_l):
                w_l_idx = k_l * n_l + i_l
                w_g_idx = (k_off + k_l) * n + (i_off + i_l)
                for c in range(rank_l):
                    W_global[w_g_idx][col_offset + c] = W_l[w_l_idx][c]
                    
        col_offset += rank_l
        
    return {"U": U_global, "V": V_global, "W": W_global}
# EVOLVE-BLOCK-END
