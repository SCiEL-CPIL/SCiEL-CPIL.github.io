import numpy as np

DIMENSIONS = [3, 4, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Target rank 78
    rank = 78
    
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    # Helper to add a block decomposition
    def add_block(n_b, m_b, p_b, i_off, j_off, k_off, decomp):
        # decomp is list of (u_idx, v_idx, w_idx, coeff)
        # u_idx in [0, n_b*m_b)
        # v_idx in [0, m_b*p_b)
        # w_idx in [0, p_b*n_b)
        for u_idx, v_idx, w_idx, coeff in decomp:
            u_global = (i_off + u_idx // m_b) * m + (j_off + u_idx % m_b)
            v_global = (j_off + v_idx // p_b) * p + (k_off + v_idx % p_b)
            w_global = (k_off + w_idx // n_b) * n + (i_off + w_idx % n_b)
            
            # We need to accumulate into the correct column
            # Since we are building column by column, we need to know which column
            # This helper assumes we are adding to a specific column or set of columns?
            # No, the decomposition defines the columns.
            # We should pass the start column index.
            pass

    # Better approach: build the full list of tuples first, then assign.
    # Or, since rank is fixed, we can just iterate.
    
    def get_strassen_222():
        # Returns list of (u, v, w) indices for the 7 Strassen terms
        # u: i*2+j, v: j*2+k, w: k*2+i
        terms = []
        # M1: (A00+A11)(B00+B11) -> C00, C11
        # U: 0,3; V: 0,3; W: 0,3
        terms.append([(0,1),(3,1)], [(0,1),(3,1)], [(0,1),(3,1)])
        # M2: (A10+A11)B00 -> C10, -C11
        # U: 2,3; V: 0; W: 1, -3 (C10 is idx 1, C11 is idx 3)
        terms.append([(2,1),(3,1)], [(0,1)], [(1,1),(3,-1)])
        # M3: A00(B01-B11) -> C01, C11
        # U: 0; V: 1, -3 (B01 is 1, B11 is 3); W: 2, 3 (C01 is 2, C11 is 3)
        terms.append([(0,1)], [(1,1),(3,-1)], [(2,1),(3,1)])
        # M4: A11(B10-B00) -> C00, C10
        # U: 3; V: 2, -0 (B10 is 2, B00 is 0); W: 0, 1 (C00 is 0, C10 is 1)
        terms.append([(3,1)], [(2,1),(0,-1)], [(0,1),(1,1)])
        # M5: (A00+A01)B11 -> -C00, C01
        # U: 0,1; V: 3; W: -0, 2
        terms.append([(0,1),(1,1)], [(3,1)], [(0,-1),(2,1)])
        # M6: (A10-A00)(B00+B01) -> C11
        # U: 2, -0; V: 0,1; W: 3
        terms.append([(2,1),(0,-1)], [(0,1),(1,1)], [(3,1)])
        # M7: (A01-A11)(B10+B11) -> C00
        # U: 1, -3; V: 2,3; W: 0
        terms.append([(1,1),(3,-1)], [(2,1),(3,1)], [(0,1)])
        
        return terms

    def get_schoolbook(n_b, m_b, p_b):
        terms = []
        for i in range(n_b):
            for j in range(m_b):
                for k in range(p_b):
                    u_idx = i * m_b + j
                    v_idx = j * p_b + k
                    w_idx = k * n_b + i
                    terms.append([[(u_idx,1)], [(v_idx,1)], [(w_idx,1)]])
        return terms

    # We will build the columns sequentially.
    # Column index `c` corresponds to one rank-1 term.
    # Each term is defined by U_col, V_col, W_col.
    
    all_terms = []
    
    # Helper to add a block's terms to all_terms
    def add_block_terms(n_b, m_b, p_b, i_off, j_off, k_off, base_terms):
        for term in base_terms:
            u_vec, v_vec, w_vec = term
            # Map local indices to global
            u_global = {}
            for idx, coeff in u_vec:
                i_local = idx // m_b
                j_local = idx % m_b
                u_global[(i_off + i_local) * m + (j_off + j_local)] = coeff
            
            v_global = {}
            for idx, coeff in v_vec:
                j_local = idx // p_b
                k_local = idx % p_b
                v_global[(j_off + j_local) * p + (k_off + k_local)] = coeff
            
            w_global = {}
            for idx, coeff in w_vec:
                k_local = idx // n_b
                i_local = idx % n_b
                w_global[(k_off + k_local) * n + (i_off + i_local)] = coeff
            
            all_terms.append((u_global, v_global, w_global))

    # 1. A1 B1 (i_off=0, j_off=0, k_off=0)
    # A1 is 3x2, B1 is 2x7
    # Split A1 into 2x2 (i_off=0) and 1x2 (i_off=2)
    
    # 1a. A1_top (2x2) * B1 (2x7)
    # B1 split into 2x4 (k_off=0) and 2x3 (k_off=4)
    
    # 1a1. A1_top * B1_left (2x2 * 2x4)
    # B1_left split into 2x2 (k_off=0) and 2x2 (k_off=2)
    # A1_top * B1_left_left (2x2 * 2x2) -> Strassen
    add_block_terms(2, 2, 2, 0, 0, 0, get_strassen_222())
    # A1_top * B1_left_right (2x2 * 2x2) -> Strassen
    add_block_terms(2, 2, 2, 0, 0, 2, get_strassen_222())
    
    # 1a2. A1_top * B1_right (2x2 * 2x3)
    # B1_right split into 2x2 (k_off=4) and 2x1 (k_off=6)
    # A1_top * B1_right_left (2x2 * 2x2) -> Strassen
    add_block_terms(2, 2, 2, 0, 0, 4, get_strassen_222())
    # A1_top * B1_right_right (2x2 * 2x1) -> Schoolbook
    add_block_terms(2, 2, 1, 0, 0, 6, get_schoolbook(2, 2, 1))
    
    # 1b. A1_bot (1x2) * B1 (2x7)
    # Schoolbook
    add_block_terms(1, 2, 7, 2, 0, 0, get_schoolbook(1, 2, 7))
    
    # 2. A2 B2 (i_off=0, j_off=2, k_off=0)
    # A2 is 3x2, B2 is 2x7
    # Split A2 into 2x2 (i_off=0) and 1x2 (i_off=2)
    
    # 2a. A2_top (2x2) * B2 (2x7)
    # B2 split into 2x4 (k_off=0) and 2x3 (k_off=4)
    
    # 2a1. A2_top * B2_left (2x2 * 2x4)
    # B2_left split into 2x2 (k_off=0) and 2x2 (k_off=2)
    # A2_top * B2_left_left (2x2 * 2x2) -> Strassen
    add_block_terms(2, 2, 2, 0, 2, 0, get_strassen_222())
    # A2_top * B2_left_right (2x2 * 2x2) -> Strassen
    add_block_terms(2, 2, 2, 0, 2, 2, get_strassen_222())
    
    # 2a2. A2_top * B2_right (2x2 * 2x3)
    # B2_right split into 2x2 (k_off=4) and 2x1 (k_off=6)
    # A2_top * B2_right_left (2x2 * 2x2) -> Strassen
    add_block_terms(2, 2, 2, 0, 2, 4, get_strassen_222())
    # A2_top * B2_right_right (2x2 * 2x1) -> Schoolbook
    add_block_terms(2, 2, 1, 0, 2, 6, get_schoolbook(2, 2, 1))
    
    # 2b. A2_bot (1x2) * B2 (2x7)
    # Schoolbook
    add_block_terms(1, 2, 7, 2, 2, 0, get_schoolbook(1, 2, 7))
    
    # Fill U, V, W
    for c, (u_global, v_global, w_global) in enumerate(all_terms):
        for idx, coeff in u_global.items():
            U[idx, c] = coeff
        for idx, coeff in v_global.items():
            V[idx, c] = coeff
        for idx, coeff in w_global.items():
            W[idx, c] = coeff
            
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
