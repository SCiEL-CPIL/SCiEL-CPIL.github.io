import time

DIMENSIONS = [3, 5, 6]

def construct():
    start_time = time.time()
    deadline = 120.0
    
    n, m, p = DIMENSIONS
    # Rank 84 decomposition
    rank = 84
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Strassen terms for 2x2x2
    # u, v, w are flattened 2x2 matrices
    # Indices: 00, 01, 10, 11
    strassen_terms = [
        # M1
        ([1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, 1]), # C00, C11
        # M2
        ([0, 0, 1, 1], [1, 0, 0, 0], [0, 1, 0, 0]), # C10 (k=1, i=0 -> idx 1)
        # M3
        ([1, 0, 0, 0], [0, 1, 0, -1], [0, 0, 1, 0]), # C01 (k=1, i=0? No C01 is i=0, k=1 -> idx 1? No W[k*n+i]. k=1, i=0 -> 1*2+0=2? No.
        # Let's re-verify W index. W[k*n+i].
        # C00: k=0, i=0 -> 0
        # C01: k=1, i=0 -> 2
        # C10: k=0, i=1 -> 1
        # C11: k=1, i=1 -> 3
        # So C01 is idx 2. C10 is idx 1.
        # My M2 w was [0, 1, 0, 0] -> idx 1. Correct for C10.
        # My M3 w was [0, 0, 1, 0] -> idx 2. Correct for C01.
        # M4
        ([0, 0, 0, 1], [0, 0, 1, -1], [1, 1, 0, 0]), # C00 (idx 0) and C10 (idx 1)
        # M5
        ([1, 1, 0, 0], [0, 0, 0, 1], [-1, 0, 1, 0]), # C00 (idx 0) and C01 (idx 2)
        # M6
        ([-1, 0, 1, 0], [1, 1, 0, 0], [0, 0, 0, 1]), # C11 (idx 3)
        # M7
        ([0, 1, 0, -1], [0, 0, 1, 1], [1, 0, 0, -1])  # C00 (idx 0) and C01 (idx 2)
    ]
    
    # Helper to add a block
    def add_block(i_off, j_off, k_off, n_b, m_b, p_b, terms):
        nonlocal col
        for u, v, w in terms:
            for i in range(n_b):
                for j in range(m_b):
                    if u[i * m_b + j] != 0:
                        U[(i_off + i) * m + (j_off + j)][col] += u[i * m_b + j]
            for j in range(m_b):
                for k in range(p_b):
                    if v[j * p_b + k] != 0:
                        V[(j_off + j) * p + (k_off + k)][col] += v[j * p_b + k]
            for k in range(p_b):
                for i in range(n_b):
                    if w[k * n_b + i] != 0:
                        W[(k_off + k) * n + (i_off + i)][col] += w[k * n_b + i]
            col += 1
            if time.time() > deadline - 5:
                return False
        return True

    # 2x5x6 part (i=0,1)
    # A cols: 2, 2, 1
    # B rows: 2, 2, 1, cols: 2, 2, 2
    a_col_sizes = [2, 2, 1]
    b_row_sizes = [2, 2, 1]
    b_col_sizes = [2, 2, 2]
    
    a_j_off = 0
    for b in range(3):
        b_row_off = 0
        for c in range(3):
            b_col_off = 0
            # Actually offsets are cumulative
            pass
        
    # Let's just compute offsets
    a_j_offsets = [0, 2, 4]
    b_r_offsets = [0, 2, 4]
    b_c_offsets = [0, 2, 4]
    
    for b in range(3):
        for c in range(3):
            n_b = 2
            m_b = a_col_sizes[b]
            p_b = b_col_sizes[c]
            
            i_off = 0
            j_off = a_j_offsets[b]
            k_off = b_c_offsets[c]
            
            if m_b == 2 and p_b == 2:
                # 2x2x2 Strassen
                if not add_block(i_off, j_off, k_off, n_b, m_b, p_b, strassen_terms):
                    return {"U": U, "V": V, "W": W}
            elif m_b == 1 and p_b == 2:
                # 2x1x2 Schoolbook
                # Terms:
                # A00 B00 -> C00
                # A00 B01 -> C01
                # A10 B00 -> C10
                # A10 B01 -> C11
                terms = [
                    ([1, 0], [1, 0], [1, 0, 0, 0]), # C00
                    ([1, 0], [0, 1], [0, 0, 1, 0]), # C01 (idx 2)
                    ([0, 1], [1, 0], [0, 1, 0, 0]), # C10 (idx 1)
                    ([0, 1], [0, 1], [0, 0, 0, 1])  # C11 (idx 3)
                ]
                if not add_block(i_off, j_off, k_off, n_b, m_b, p_b, terms):
                    return {"U": U, "V": V, "W": W}
            
    # 1x5x6 part (i=2)
    for b in range(3):
        for c in range(3):
            n_b = 1
            m_b = a_col_sizes[b]
            p_b = b_col_sizes[c]
            
            i_off = 2
            j_off = a_j_offsets[b]
            k_off = b_c_offsets[c]
            
            if m_b == 2 and p_b == 2:
                # 1x2x2 Schoolbook
                # A00 B00 -> C00
                # A00 B01 -> C01
                # A01 B00 -> C10? No C is 1x2. C0, C1.
                # C0 = A00 B00 + A01 B10
                # C1 = A00 B01 + A01 B11
                # Terms:
                # A00 B00 -> C0
                # A00 B01 -> C1
                # A01 B00 -> C0
                # A01 B01 -> C1
                terms = [
                    ([1, 0], [1, 0], [1, 0]), # C0
                    ([1, 0], [0, 1], [0, 1]), # C1
                    ([0, 1], [1, 0], [1, 0]), # C0
                    ([0, 1], [0, 1], [0, 1])  # C1
                ]
                if not add_block(i_off, j_off, k_off, n_b, m_b, p_b, terms):
                    return {"U": U, "V": V, "W": W}
            elif m_b == 1 and p_b == 2:
                # 1x1x2 Schoolbook
                # A00 B00 -> C0
                # A00 B01 -> C1
                terms = [
                    ([1], [1, 0], [1, 0]),
                    ([1], [0, 1], [0, 1])
                ]
                if not add_block(i_off, j_off, k_off, n_b, m_b, p_b, terms):
                    return {"U": U, "V": V, "W": W}
                    
    return {"U": U, "V": V, "W": W}
