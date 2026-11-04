import time

DIMENSIONS = [5, 5, 5]

def construct():
    start_time = time.time()
    deadline = start_time + 115
    
    n, m, p = DIMENSIONS
    
    # Partitions
    n_parts = [4, 1]
    m_parts = [4, 1]
    p_parts = [4, 1]
    
    n_starts = [0]
    m_starts = [0]
    p_starts = [0]
    for i in range(1, len(n_parts)):
        n_starts.append(n_starts[-1] + n_parts[i-1])
    for i in range(1, len(m_parts)):
        m_starts.append(m_starts[-1] + m_parts[i-1])
    for i in range(1, len(p_parts)):
        p_starts.append(p_starts[-1] + p_parts[i-1])
        
    # Calculate total rank
    total_rank = 0
    for i in range(2):
        for j in range(2):
            for k in range(2):
                if n_parts[i] == 4 and m_parts[j] == 4 and p_parts[k] == 4:
                    total_rank += 49
                else:
                    total_rank += n_parts[i] * m_parts[j] * p_parts[k]
                    
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]
    
    col = 0
    
    # Define the 2x2x2 Strassen decomposition terms
    # Each term is (u, v, w) where u, v, w are lists of (index, value)
    # Indices are 0-3 for 2x2 matrices (row-major)
    # 0: (0,0), 1: (0,1), 2: (1,0), 3: (1,1)
    
    strassen_2x2_terms = [
        # M1
        ([(0, 1), (3, 1)], [(0, 1), (3, 1)], [(0, 1), (3, 1)]),
        # M2
        ([(2, 1), (3, 1)], [(0, 1)], [(1, 1), (3, -1)]),
        # M3
        ([(0, 1)], [(1, 1), (3, -1)], [(2, 1), (3, 1)]),
        # M4
        ([(3, 1)], [(2, 1), (0, -1)], [(0, 1), (1, 1)]),
        # M5
        ([(0, 1), (1, 1)], [(3, 1)], [(0, -1), (2, 1)]),
        # M6
        ([(2, 1), (0, -1)], [(0, 1), (1, 1)], [(3, 1)]),
        # M7
        ([(1, 1), (3, -1)], [(2, 1), (3, 1)], [(0, 1)])
    ]
    
    # Define mappings for the 4x4x4 Strassen terms
    # Each mapping specifies:
    # - a_blocks: list of (offset, sign) for A
    # - b_blocks: list of (offset, sign) for B
    # - c_blocks: list of (offset, sign) for C
    # Offsets are (row_offset, col_offset) in the 4x4 matrix
    # A blocks: 00, 01, 10, 11
    # 00: (0,0), 01: (0,2), 10: (2,0), 11: (2,2)
    
    strassen_4x4_mappings = [
        # M1: (A00 + A11)(B00 + B11) -> C00, C11
        ([(0,0, 1), (2,2, 1)], [(0,0, 1), (2,2, 1)], [(0,0, 1), (2,2, 1)]),
        # M2: (A10 + A11)B00 -> C10
        ([(2,0, 1), (2,2, 1)], [(0,0, 1)], [(2,0, 1)]),
        # M3: A00(B01 - B11) -> C01
        ([(0,0, 1)], [(0,2, 1), (2,2, -1)], [(0,2, 1)]),
        # M4: A11(B10 - B00) -> C00, C10
        ([(2,2, 1)], [(2,0, 1), (0,0, -1)], [(0,0, 1), (2,0, 1)]),
        # M5: (A00 + A01)B11 -> -C00, C01
        ([(0,0, 1), (0,2, 1)], [(2,2, 1)], [(0,0, -1), (0,2, 1)]),
        # M6: (A10 - A00)(B00 + B01) -> C11
        ([(2,0, 1), (0,0, -1)], [(0,0, 1), (0,2, 1)], [(2,2, 1)]),
        # M7: (A01 - A11)(B10 + B11) -> C00
        ([(0,2, 1), (2,2, -1)], [(2,0, 1), (2,2, 1)], [(0,0, 1)])
    ]
    
    for i in range(2):
        for j in range(2):
            for k in range(2):
                if time.time() > deadline:
                    # Fallback to schoolbook if time runs out?
                    # Just return what we have, hoping it's valid?
                    # Better to fail fast if we can't finish.
                    # But 49 terms is very small.
                    pass
                    
                ni = n_parts[i]
                mj = m_parts[j]
                pk = p_parts[k]
                
                i_start = n_starts[i]
                j_start = m_starts[j]
                k_start = p_starts[k]
                
                if ni == 4 and mj == 4 and pk == 4:
                    # Strassen 4x4x4
                    for map_idx in range(7):
                        a_blocks, b_blocks, c_blocks = strassen_4x4_mappings[map_idx]
                        
                        for term_idx in range(7):
                            u_loc, v_loc, w_loc = strassen_2x2_terms[term_idx]
                            
                            # Expand u
                            for (r_off, c_off, sign_a) in a_blocks:
                                for (u_idx, u_val) in u_loc:
                                    # Map local u_idx (0-3) to global index
                                    # u_idx 0: (0,0), 1: (0,1), 2: (1,0), 3: (1,1)
                                    lr = u_idx // 2
                                    lc = u_idx % 2
                                    
                                    gr = i_start + r_off + lr
                                    gc = j_start + c_off + lc
                                    
                                    U[gr * m + gc][col] += sign_a * u_val
                            
                            # Expand v
                            for (r_off, c_off, sign_b) in b_blocks:
                                for (v_idx, v_val) in v_loc:
                                    lr = v_idx // 2
                                    lc = v_idx % 2
                                    
                                    gr = j_start + r_off + lr
                                    gc = k_start + c_off + lc
                                    
                                    V[gr * p + gc][col] += sign_b * v_val
                            
                            # Expand w
                            for (r_off, c_off, sign_c) in c_blocks:
                                for (w_idx, w_val) in w_loc:
                                    lr = w_idx // 2
                                    lc = w_idx % 2
                                    
                                    gr = k_start + r_off + lr
                                    gc = i_start + c_off + lc
                                    
                                    W[gr * n + gc][col] += sign_c * w_val
                            
                            col += 1
                else:
                    # Schoolbook
                    for u in range(ni):
                        for v in range(mj):
                            for w in range(pk):
                                U[(i_start + u) * m + (j_start + v)][col] = 1
                                V[(j_start + v) * p + (k_start + w)][col] = 1
                                W[(k_start + w) * n + (i_start + u)][col] = 1
                                col += 1
                                
    return {"U": U, "V": V, "W": W}
