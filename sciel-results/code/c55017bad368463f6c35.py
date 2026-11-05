"""
Optimized rank-65 decomposition for 4x4 * 4x5 matrix multiplication.
Uses two-level Strassen (rank 49) for the 4x4x4 part and schoolbook (rank 16) for the 4x4x1 part.
"""

import time

def construct():
    n, m, p = 4, 4, 5
    deadline = time.time() + 140
    
    # Total rank is 49 (Strassen) + 16 (Schoolbook) = 65
    rank = 65
    U = [[0] * rank for _ in range(n * m)]  # 16 x 65
    V = [[0] * rank for _ in range(m * p)]  # 20 x 65
    W = [[0] * rank for _ in range(p * n)]  # 20 x 65
    
    def add_term(col, u_list, v_list, w_list):
        for i, j, c in u_list:
            if c != 0:
                U[i * m + j][col] += c
        for j, k, c in v_list:
            if c != 0:
                V[j * p + k][col] += c
        for k, i, c in w_list:
            if c != 0:
                W[k * n + i][col] += c
    
    col = 0
    
    # --- Part 1: 4x4 * 4x4 using two-level Strassen (rank 49) ---
    
    # Helper to get global indices for blocks
    def a_block_elem(block_idx, r, c):
        if block_idx == 0: return (r, c)
        elif block_idx == 1: return (r, 2 + c)
        elif block_idx == 2: return (2 + r, c)
        else: return (2 + r, 2 + c)
    
    def b_block_elem(block_idx, r, c):
        if block_idx == 0: return (r, c)
        elif block_idx == 1: return (r, 2 + c)
        elif block_idx == 2: return (2 + r, c)
        else: return (2 + r, 2 + c)
    
    def c_block_elem(block_idx, r, c):
        if block_idx == 0: return (r, c)
        elif block_idx == 1: return (r, 2 + c)
        elif block_idx == 2: return (2 + r, c)
        else: return (2 + r, 2 + c)
    
    # Level 1 Strassen: 7 products
    level1_products = [
        {'p_blocks': [(0, 1), (3, 1)], 'q_blocks': [(0, 1), (3, 1)], 'c_contrib': {0: 1, 3: 1}},
        {'p_blocks': [(2, 1), (3, 1)], 'q_blocks': [(0, 1)], 'c_contrib': {2: 1, 3: -1}},
        {'p_blocks': [(0, 1)], 'q_blocks': [(1, 1), (3, -1)], 'c_contrib': {1: 1, 3: 1}},
        {'p_blocks': [(3, 1)], 'q_blocks': [(2, 1), (0, -1)], 'c_contrib': {0: 1, 2: 1}},
        {'p_blocks': [(0, 1), (1, 1)], 'q_blocks': [(3, 1)], 'c_contrib': {0: -1, 1: 1}},
        {'p_blocks': [(2, 1), (0, -1)], 'q_blocks': [(0, 1), (1, 1)], 'c_contrib': {3: 1}},
        {'p_blocks': [(1, 1), (3, -1)], 'q_blocks': [(2, 1), (3, 1)], 'c_contrib': {0: 1}},
    ]
    
    # Level 2 Strassen: 7 products for 2x2 * 2x2
    level2_strassen = [
        {'p_elems': [(0, 1), (3, 1)], 'q_elems': [(0, 1), (3, 1)], 'c_contrib': {0: 1, 3: 1}},
        {'p_elems': [(2, 1), (3, 1)], 'q_elems': [(0, 1)], 'c_contrib': {2: 1, 3: -1}},
        {'p_elems': [(0, 1)], 'q_elems': [(1, 1), (3, -1)], 'c_contrib': {1: 1, 3: 1}},
        {'p_elems': [(3, 1)], 'q_elems': [(2, 1), (0, -1)], 'c_contrib': {0: 1, 2: 1}},
        {'p_elems': [(0, 1), (1, 1)], 'q_elems': [(3, 1)], 'c_contrib': {0: -1, 1: 1}},
        {'p_elems': [(2, 1), (0, -1)], 'q_elems': [(0, 1), (1, 1)], 'c_contrib': {3: 1}},
        {'p_elems': [(1, 1), (3, -1)], 'q_elems': [(2, 1), (3, 1)], 'c_contrib': {0: 1}},
    ]
    
    for l1_idx, l1_prod in enumerate(level1_products):
        for l2_idx, l2_prod in enumerate(level2_strassen):
            if time.time() > deadline:
                return {"U": U, "V": V, "W": W} # Fallback, though unlikely
            
            u_list = []
            v_list = []
            w_list = []
            
            # Build u_list
            for p_block_idx, p_block_coeff in l1_prod['p_blocks']:
                for p_elem_idx, p_elem_coeff in l2_prod['p_elems']:
                    r = p_elem_idx // 2
                    c = p_elem_idx % 2
                    gi, gj = a_block_elem(p_block_idx, r, c)
                    coeff = p_block_coeff * p_elem_coeff
                    u_list.append((gi, gj, coeff))
            
            # Build v_list
            for q_block_idx, q_block_coeff in l1_prod['q_blocks']:
                for q_elem_idx, q_elem_coeff in l2_prod['q_elems']:
                    r = q_elem_idx // 2
                    c = q_elem_idx % 2
                    gj, gk = b_block_elem(q_block_idx, r, c)
                    coeff = q_block_coeff * q_elem_coeff
                    v_list.append((gj, gk, coeff))
            
            # Build w_list
            for si_elem_idx, si_coeff in l2_prod['c_contrib'].items():
                si_r = si_elem_idx // 2
                si_c = si_elem_idx % 2
                for l1_c_block, l1_c_coeff in l1_prod['c_contrib'].items():
                    gi, gj = c_block_elem(l1_c_block, si_r, si_c)
                    coeff = l1_c_coeff * si_coeff
                    w_list.append((gj, gi, coeff))
            
            add_term(col, u_list, v_list, w_list)
            col += 1
    
    # --- Part 2: 4x4 * 4x1 using schoolbook (rank 16) ---
    for i in range(n):
        for j in range(m):
            if time.time() > deadline:
                return {"U": U, "V": V, "W": W}
            u_list = [(i, j, 1)]
            v_list = [(j, 4, 1)]
            w_list = [(4, i, 1)]
            add_term(col, u_list, v_list, w_list)
            col += 1
    
    assert col == rank, f"Expected {rank} terms, got {col}"
    
    return {"U": U, "V": V, "W": W}
