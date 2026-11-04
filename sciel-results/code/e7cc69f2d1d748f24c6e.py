<<<<<<< SEARCH
    # Try all possible splits and pick the best
    best = None
    best_rank = m * n * p  # schoolbook
    
    for m1 in range(1, m):
        m2 = m - m1
        for n1 in range(1, n):
            n2 = n - n1
            for p1 in range(1, p):
                p2 = p - p1
                try:
                    # 8 sub-decompositions
                    U1, V1, W1 = _decompose(m1, n1, p1)
                    U2, V2, W2 = _decompose(m1, n1, p2)
                    U3, V3, W3 = _decompose(m1, n2, p1)
                    U4, V4, W4 = _decompose(m1, n2, p2)
                    U5, V5, W5 = _decompose(m2, n1, p1)
                    U6, V6, W6 = _decompose(m2, n1, p2)
                    U7, V7, W7 = _decompose(m2, n2, p1)
                    U8, V8, W8 = _decompose(m2, n2, p2)
                    
                    total_rank = (len(U1) + len(U2) + len(U3) + len(U4) +
                                 len(U5) + len(U6) + len(U7) + len(U8))
                    if total_rank < best_rank:
                        best_rank = total_rank
                        # Build combined U, V, W
                        U_new = []
                        V_new = []
                        W_new = []
                        
                        def add_decomp(U_s, V_s, W_s, m_i, n_i, p_i, offset_m, offset_n, offset_p):
                            for r in range(len(U_s)):
                                u_row = [0] * (m * n)
                                v_row = [0] * (n * p)
                                w_row = [0] * (p * m)
                                for ii in range(m_i):
                                    for jj in range(n_i):
                                        u_row[(offset_m + ii) * n + (offset_n + jj)] = U_s[r][ii * n_i + jj]
                                for jj in range(n_i):
                                    for kk in range(p_i):
                                        v_row[(offset_n + jj) * p + (offset_p + kk)] = V_s[r][jj * p_i + kk]
                                for kk in range(p_i):
                                    for ii in range(m_i):
                                        w_row[(offset_p + kk) * m + (offset_m + ii)] = W_s[r][kk * m_i + ii]
                                U_new.append(u_row)
                                V_new.append(v_row)
                                W_new.append(w_row)
                        
                        # A11*B11: m1×n1 * n1×p1
                        add_decomp(U1, V1, W1, m1, n1, p1, 0, 0, 0)
                        # A12*B21: m1×n2 * n2×p1
                        add_decomp(U3, V3, W3, m1, n2, p1, 0, n1, 0)
                        # A11*B12: m1×n1 * n1×p2
                        add_decomp(U2, V2, W2, m1, n1, p2, 0, 0, p1)
                        # A12*B22: m1×n2 * n2×p2
                        add_decomp(U4, V4, W4, m1, n2, p2, 0, n1, p1)
                        # A21*B11: m2×n1 * n1×p1
                        add_decomp(U5, V5, W5, m2, n1, p1, m1, 0, 0)
                        # A21*B12: m2×n1 * n1×p2
                        add_decomp(U6, V6, W6, m2, n1, p2, m1, 0, p1)
                        # A22*B21: m2×n2 * n2×p1
                        add_decomp(U7, V7, W7, m2, n2, p1, m1, n1, 0)
                        # A22*B22: m2×n2 * n2×p2
                        add_decomp(U8, V8, W8, m2, n2, p2, m1, n1, p1)
                        
                        best = (U_new, V_new, W_new)
                except TimeoutError:
                    raise
                except Exception:
                    continue
    
    if best is not None:
        return best
    else:
        # Schoolbook
        rank = m * n * p
        U = [[0] * rank for _ in range(m * n)]
        V = [[0] * rank for _ in range(n * p)]
        W = [[0] * rank for _ in range(p * m)]
        col = 0
        for i in range(m):
            for j in range(n):
                for k in range(p):
                    U[i * n + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * m + i][col] = 1
                    col += 1
        return (U, V, W)
=======
    # Special case: 4x5x5 with rank-85 block decomposition
    if m == 4 and n == 5 and p == 5:
        return _block_decomp_4x5x5()
    
    # Try all possible splits and pick the best
    best = None
    best_rank = m * n * p  # schoolbook
    
    for m1 in range(1, m):
        m2 = m - m1
        for n1 in range(1, n):
            n2 = n - n1
            for p1 in range(1, p):
                p2 = p - p1
                try:
                    # 8 sub-decompositions
                    U1, V1, W1 = _decompose(m1, n1, p1)
                    U2, V2, W2 = _decompose(m1, n1, p2)
                    U3, V3, W3 = _decompose(m1, n2, p1)
                    U4, V4, W4 = _decompose(m1, n2, p2)
                    U5, V5, W5 = _decompose(m2, n1, p1)
                    U6, V6, W6 = _decompose(m2, n1, p2)
                    U7, V7, W7 = _decompose(m2, n2, p1)
                    U8, V8, W8 = _decompose(m2, n2, p2)
                    
                    total_rank = (len(U1) + len(U2) + len(U3) + len(U4) +
                                 len(U5) + len(U6) + len(U7) + len(U8))
                    if total_rank < best_rank:
                        best_rank = total_rank
                        # Build combined U, V, W
                        U_new = []
                        V_new = []
                        W_new = []
                        
                        def add_decomp(U_s, V_s, W_s, m_i, n_i, p_i, offset_m, offset_n, offset_p):
                            for r in range(len(U_s)):
                                u_row = [0] * (m * n)
                                v_row = [0] * (n * p)
                                w_row = [0] * (p * m)
                                for ii in range(m_i):
                                    for jj in range(n_i):
                                        u_row[(offset_m + ii) * n + (offset_n + jj)] = U_s[r][ii * n_i + jj]
                                for jj in range(n_i):
                                    for kk in range(p_i):
                                        v_row[(offset_n + jj) * p + (offset_p + kk)] = V_s[r][jj * p_i + kk]
                                for kk in range(p_i):
                                    for ii in range(m_i):
                                        w_row[(offset_p + kk) * m + (offset_m + ii)] = W_s[r][kk * m_i + ii]
                                U_new.append(u_row)
                                V_new.append(v_row)
                                W_new.append(w_row)
                        
                        # A11*B11: m1×n1 * n1×p1
                        add_decomp(U1, V1, W1, m1, n1, p1, 0, 0, 0)
                        # A12*B21: m1×n2 * n2×p1
                        add_decomp(U3, V3, W3, m1, n2, p1, 0, n1, 0)
                        # A11*B12: m1×n1 * n1×p2
                        add_decomp(U2, V2, W2, m1, n1, p2, 0, 0, p1)
                        # A12*B22: m1×n2 * n2×p2
                        add_decomp(U4, V4, W4, m1, n2, p2, 0, n1, p1)
                        # A21*B11: m2×n1 * n1×p1
                        add_decomp(U5, V5, W5, m2, n1, p1, m1, 0, 0)
                        # A21*B12: m2×n1 * n1×p2
                        add_decomp(U6, V6, W6, m2, n1, p2, m1, 0, p1)
                        # A22*B21: m2×n2 * n2×p1
                        add_decomp(U7, V7, W7, m2, n2, p1, m1, n1, 0)
                        # A22*B22: m2×n2 * n2×p2
                        add_decomp(U8, V8, W8, m2, n2, p2, m1, n1, p1)
                        
                        best = (U_new, V_new, W_new)
                except TimeoutError:
                    raise
                except Exception:
                    continue
    
    if best is not None:
        return best
    else:
        # Schoolbook
        rank = m * n * p
        U = [[0] * rank for _ in range(m * n)]
        V = [[0] * rank for _ in range(n * p)]
        W = [[0] * rank for _ in range(p * m)]
        col = 0
        for i in range(m):
            for j in range(n):
                for k in range(p):
                    U[i * n + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * m + i][col] = 1
                    col += 1
        return (U, V, W)
>>>>>>> REPLACE

