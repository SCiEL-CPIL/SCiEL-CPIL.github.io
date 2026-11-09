DIMENSIONS = [5, 5, 5]

def _get_strassen_2x2x2():
    # Returns U, V, W as lists of lists for 2x2x2 Strassen
    # U: 7 x 2 x 2 (flattened to 7 x 4)
    # V: 7 x 2 x 2 (flattened to 7 x 4)
    # W: 7 x 2 x 2 (flattened to 7 x 4)
    
    U = []
    V = []
    W = []
    
    # M0: (A00+A11)(B00+B11)
    U.append([1, 0, 0, 1]) # A00, A01, A10, A11
    V.append([1, 0, 0, 1]) # B00, B01, B10, B11
    W.append([1, 0, 0, 1]) # C00, C01, C10, C11 (W[k,i] -> C[i,k])
    
    # M1: (A10+A11)B00
    U.append([0, 0, 1, 1])
    V.append([1, 0, 0, 0])
    W.append([0, 1, 0, -1]) # C10, C11
    
    # M2: A00(B01-B11)
    U.append([1, 0, 0, 0])
    V.append([0, 1, 0, -1])
    W.append([0, 1, 0, 1]) # C01, C11
    
    # M3: A11(B10-B00)
    U.append([0, 0, 0, 1])
    V.append([-1, 0, 1, 0])
    W.append([1, 0, 1, 0]) # C00, C10
    
    # M4: (A00+A01)B11
    U.append([1, 1, 0, 0])
    V.append([0, 0, 0, 1])
    W.append([-1, 0, 1, 0]) # C00, C01
    
    # M5: (A10-A00)(B00+B01)
    U.append([-1, 0, 1, 0])
    V.append([1, 1, 0, 0])
    W.append([0, 0, 0, 1]) # C11
    
    # M6: (A01-A11)(B10+B11)
    U.append([0, 1, 0, -1])
    V.append([0, 0, 1, 1])
    W.append([1, 0, 0, 0]) # C00
    
    return U, V, W

def _get_schoolbook(n, m, p):
    U = []
    V = []
    W = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u = [0] * (n * m)
                u[i * m + j] = 1
                v = [0] * (m * p)
                v[j * p + k] = 1
                w = [0] * (p * n)
                w[k * n + i] = 1
                U.append(u)
                V.append(v)
                W.append(w)
    return U, V, W

def construct():
    n, m, p = DIMENSIONS
    # Partition indices into blocks of size 2, 2, 1
    # Block 0: 0, 1
    # Block 1: 2, 3
    # Block 2: 4
    partition = [2, 2, 1]
    block_starts = [0, 2, 4]
    
    U_global = []
    V_global = []
    W_global = []
    
    for b_i in range(3):
        for b_j in range(3):
            for b_k in range(3):
                n_sub = partition[b_i]
                m_sub = partition[b_j]
                p_sub = partition[b_k]
                
                # Get local decomposition
                if n_sub == 2 and m_sub == 2 and p_sub == 2:
                    U_loc, V_loc, W_loc = _get_strassen_2x2x2()
                else:
                    U_loc, V_loc, W_loc = _get_schoolbook(n_sub, m_sub, p_sub)
                
                # Map local indices to global
                # Local i, j, k are 0..size-1
                # Global i = block_starts[b_i] + i
                # Global j = block_starts[b_j] + j
                # Global k = block_starts[b_k] + k
                
                for r in range(len(U_loc)):
                    u_g = [0] * (n * m)
                    v_g = [0] * (m * p)
                    w_g = [0] * (p * n)
                    
                    for li in range(n_sub):
                        for lj in range(m_sub):
                            val = U_loc[r][li * m_sub + lj]
                            if val != 0:
                                gi = block_starts[b_i] + li
                                gj = block_starts[b_j] + lj
                                u_g[gi * m + gj] = val
                                
                    for lj in range(m_sub):
                        for lk in range(p_sub):
                            val = V_loc[r][lj * p_sub + lk]
                            if val != 0:
                                gj = block_starts[b_j] + lj
                                gk = block_starts[b_k] + lk
                                v_g[gj * p + gk] = val
                                
                    for lk in range(p_sub):
                        for li in range(n_sub):
                            val = W_loc[r][lk * n_sub + li]
                            if val != 0:
                                gk = block_starts[b_k] + lk
                                gi = block_starts[b_i] + li
                                w_g[gk * n + gi] = val
                                
                    U_global.append(u_g)
                    V_global.append(v_g)
                    W_global.append(w_g)
                    
    return {"U": U_global, "V": V_global, "W": W_global}
