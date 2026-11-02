def construct():
    n, m, p = DIMENSIONS
    u_dim = n * m  # 4
    v_dim = m * p  # 8
    w_dim = p * n  # 8
    
    rank = 14
    U = [[0] * rank for _ in range(u_dim)]
    V = [[0] * rank for _ in range(v_dim)]
    W = [[0] * rank for _ in range(w_dim)]
    
    # Define the 7 Strassen terms for 2x2x2
    # Each term: (u_entries, v_entries, w_entries)
    # u_entries: list of (i, j, val)
    # v_entries: list of (j, k, val)
    # w_entries: list of (k, i, val)
    
    base_terms = [
        # 1
        ([(0,0,1), (1,1,1)], [(0,0,1), (1,1,1)], [(0,0,1), (1,1,1)]),
        # 2
        ([(1,0,1), (1,1,1)], [(0,0,1)], [(0,1,1), (1,1,-1)]),
        # 3
        ([(0,0,1)], [(0,1,1), (1,1,-1)], [(1,0,1), (1,1,1)]),
        # 4
        ([(1,1,1)], [(1,0,1), (0,0,-1)], [(0,0,1), (0,1,1)]),
        # 5
        ([(0,0,1), (0,1,1)], [(1,1,1)], [(1,0,1), (0,0,-1)]),
        # 6
        ([(1,0,1), (0,0,-1)], [(0,0,1), (0,1,1)], [(1,1,1)]),
        # 7
        ([(0,1,1), (1,1,-1)], [(1,0,1), (1,1,1)], [(0,0,1)])
    ]
    
    for r in range(14):
        base_idx = r % 7
        block = r // 7
        u_entries, v_entries, w_entries = base_terms[base_idx]
        
        # Fill U
        for i, j, val in u_entries:
            u_idx = i * m + j
            U[u_idx][r] += val
            
        # Fill V
        for j, k, val in v_entries:
            k_new = k + block * 2
            v_idx = j * p + k_new
            V[v_idx][r] += val
            
        # Fill W
        for k, i, val in w_entries:
            k_new = k + block * 2
            w_idx = k_new * n + i
            W[w_idx][r] += val
            
    return {"U": U, "V": V, "W": W}
