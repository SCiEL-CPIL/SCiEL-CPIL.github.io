import time

DIMENSIONS = [2, 3, 4]

def construct():
    n, m, p = DIMENSIONS
    # We use rank 16 decomposition
    rank = 16
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Define Strassen terms for 2x2x2
    # Local indices i,j,k in {0,1}
    # U index: 2*i + j
    # V index: 2*j + k
    # W index: 2*k + i
    
    strassen_terms = [
        # (U_coeffs, V_coeffs, W_coeffs)
        # M1
        ([1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, 1]), # W: C00(1), C11(1) -> W[0]=1, W[3]=1
        # M2
        ([0, 0, 1, 1], [1, 0, 0, 0], [0, 1, 0, -1]), # W: C10(1), C11(-1) -> W[1]=1, W[3]=-1
        # M3
        ([1, 0, 0, 0], [0, 1, 0, -1], [0, 0, 1, 1]), # W: C01(1), C11(1) -> W[2]=1, W[3]=1
        # M4
        ([0, 0, 0, 1], [0, 0, 1, -1], [1, 1, 0, 0]), # W: C00(1), C10(1) -> W[0]=1, W[1]=1
        # M5
        ([1, 1, 0, 0], [0, 0, 0, 1], [-1, 0, 1, 0]), # W: C00(-1), C01(1) -> W[0]=-1, W[2]=1
        # M6
        ([-1, 0, 1, 0], [1, 1, 0, 0], [0, 0, 0, 1]), # W: C11(1) -> W[3]=1
        # M7
        ([0, 1, 0, -1], [0, 0, 1, 1], [1, 0, 0, 1]), # W: C00(1), C11(1) -> W[0]=1, W[3]=1
    ]
    
    # Define Rank 1 term for 2x1x2
    # Local indices i in {0,1}, j=0, k in {0,1}
    # U index: i
    # V index: k
    # W index: 2*k + i
    rank1_term = ([1, 1], [1, 1], [1, 1, 1, 1])
    
    current_col = 0
    
    # Helper to add a term
    def add_term(U_local, V_local, W_local, i_offset, j_offset, k_offset, m_local, p_local):
        nonlocal current_col
        c = current_col
        current_col += 1
        
        # Map U
        for i in range(2):
            for j in range(m_local):
                u_idx_local = i * m_local + j
                u_idx_global = (i + i_offset) * m + (j + j_offset)
                U[u_idx_global][c] += U_local[u_idx_local]
                
        # Map V
        for j in range(m_local):
            for k in range(p_local):
                v_idx_local = j * p_local + k
                v_idx_global = (j + j_offset) * p + (k + k_offset)
                V[v_idx_global][c] += V_local[v_idx_local]
                
        # Map W
        for k in range(p_local):
            for i in range(2):
                w_idx_local = k * 2 + i
                w_idx_global = (k + k_offset) * n + (i + i_offset)
                W[w_idx_global][c] += W_local[w_idx_local]
                
    # Block 1: k in {0, 1}
    # Sub-block 1a: j in {0, 1}, k in {0, 1}
    for term in strassen_terms:
        add_term(term[0], term[1], term[2], i_offset=0, j_offset=0, k_offset=0, m_local=2, p_local=2)
        
    # Sub-block 1b: j=2, k in {0, 1}
    add_term(rank1_term[0], rank1_term[1], rank1_term[2], i_offset=0, j_offset=2, k_offset=0, m_local=1, p_local=2)
    
    # Block 2: k in {2, 3}
    # Sub-block 2a: j in {0, 1}, k in {2, 3}
    for term in strassen_terms:
        add_term(term[0], term[1], term[2], i_offset=0, j_offset=0, k_offset=2, m_local=2, p_local=2)
        
    # Sub-block 2b: j=2, k in {2, 3}
    add_term(rank1_term[0], rank1_term[1], rank1_term[2], i_offset=0, j_offset=2, k_offset=2, m_local=1, p_local=2)
    
    return {"U": U, "V": V, "W": W}
