<<<<<<< SEARCH
    U, V, W, current_rank = _build_initial()
    
    if not _verify(U, V, W, n, m, p):
        return _schoolbook()
    
    best_rank = current_rank
    best_U, best_V, best_W = [row[:] for row in U], [row[:] for row in V], [row[:] for row in W]
    
    # Try to reduce rank using ALS optimization
    np.random.seed(42)
=======
    U, V, W, current_rank = _build_initial()
    
    if not _verify(U, V, W, n, m, p):
        return _schoolbook()
    
    best_rank = current_rank
    best_U, best_V, best_W = [row[:] for row in U], [row[:] for row in V], [row[:] for row in W]
    
    # Try to reduce rank using ALS optimization
    np.random.seed(42)
    
    # Build the 48-rank structured decomposition for use in initialization
    U_s = np.zeros((n * m, 48), dtype=np.float64)
    V_s = np.zeros((m * p, 48), dtype=np.float64)
    W_s = np.zeros((p * n, 48), dtype=np.float64)
    
    strassen_tensors = _strassen_2x2x2()
    col = 0
    
    for block in range(3):
        k_offset = block * 2
        for (u, v, w) in strassen_tensors:
            u_map = [(0, 0), (0, 1), (1, 0), (1, 1)]
            for idx, (ii, jj) in enumerate(u_map):
                a_idx = ii * m + jj
                U_s[a_idx, col] += u[idx]
            v_map = [(0, k_offset), (0, k_offset+1), (1, k_offset), (1, k_offset+1)]
            for idx, (jj, kk) in enumerate(v_map):
                b_idx = jj * p + kk
                V_s[b_idx, col] += v[idx]
            w_map = [(k_offset, 0), (k_offset+1, 0), (k_offset, 1), (k_offset+1, 1)]
            for idx, (kk, ii) in enumerate(w_map):
                c_idx = kk * n + ii
                W_s[c_idx, col] += w[idx]
            col += 1
    
    for block in range(3):
        k_offset = block * 2
        for (u, v, w) in strassen_tensors:
            u_map = [(0, 2), (0, 3), (1, 2), (1, 3)]
            for idx, (ii, jj) in enumerate(u_map):
                a_idx = ii * m + jj
                U_s[a_idx, col] += u[idx]
            v_map = [(2, k_offset), (2, k_offset+1), (3, k_offset), (3, k_offset+1)]
            for idx, (jj, kk) in enumerate(v_map):
                b_idx = jj * p + kk
                V_s[b_idx, col] += v[idx]
            w_map = [(k_offset, 0), (k_offset+1, 0), (k_offset, 1), (k_offset+1, 1)]
            for idx, (kk, ii) in enumerate(w_map):
                c_idx = kk * n + ii
                W_s[c_idx, col] += w[idx]
            col += 1
    
    for i in range(n):
        a_idx = i * m + 4
        for k in range(p):
            b_idx = 4 * p + k
            c_idx = k * n + i
            U_s[a_idx, col] = 1.0
            V_s[b_idx, col] = 1.0
            W_s[c_idx, col] = 1.0
            col += 1
>>>>>>> REPLACE

Now let me fix the `_build_initial` function to use the correct rank of 48 instead of 54:

