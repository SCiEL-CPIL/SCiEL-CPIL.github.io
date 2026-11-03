def construct():
    n, m, p = 3, 5, 6
    terms = []  # list of (u_dict, v_dict, w_dict)
    
    # Strassen 2x2x2 terms
    strassen_U = [
        [1,0,0,1],  # M1
        [0,0,1,1],  # M2
        [1,0,0,0],  # M3
        [0,0,0,1],  # M4
        [1,1,0,0],  # M5
        [-1,0,1,0], # M6
        [0,1,0,-1], # M7
    ]
    strassen_V = [
        [1,0,0,1],  # M1
        [1,0,0,0],  # M2
        [0,1,0,-1], # M3
        [0,0,1,-1], # M4
        [0,0,0,1],  # M5
        [1,0,1,0],  # M6
        [0,1,0,1],  # M7
    ]
    strassen_W = [
        [1,0,0,1],   # M1
        [0,0,1,-1],  # M2
        [0,1,0,1],   # M3
        [1,0,1,0],   # M4
        [-1,1,0,0],  # M5
        [0,0,0,1],   # M6
        [1,0,0,0],   # M7
    ]
    
    # For each column block c in 0,1,2 (cols 2c, 2c+1)
    for c in range(3):
        k0 = 2 * c  # starting column
        
        # Top block: rows 0-1, middle split into [0,1], [2,3], [4]
        # 2x2x2 for middle [0,1]
        i0 = 0
        for j0 in [0, 2]:
            for t in range(7):
                u = {}
                v = {}
                w = {}
                for ip in range(2):
                    for jp in range(2):
                        idx_u = (i0+ip)*m + (j0+jp)
                        u[idx_u] = u.get(idx_u, 0) + strassen_U[t][ip*2+jp]
                for jp in range(2):
                    for kp in range(2):
                        idx_v = (j0+jp)*p + (k0+kp)
                        v[idx_v] = v.get(idx_v, 0) + strassen_V[t][jp*2+kp]
                for ip in range(2):
                    for kp in range(2):
                        idx_w = (k0+kp)*n + (i0+ip)
                        w[idx_w] = w.get(idx_w, 0) + strassen_W[t][ip*2+kp]
                terms.append((u, v, w))
        
        # 2x1x2 for middle [4]
        i0 = 0
        j0 = 4
        for ip in range(2):
            for kp in range(2):
                u = {(i0+ip)*m + j0: 1}
                v = {j0*p + (k0+kp): 1}
                w = {(k0+kp)*n + (i0+ip): 1}
                terms.append((u, v, w))
        
        # Bottom block: row 2, middle split into [0,1], [2,3], [4]
        # 1x2x2 for middle [0,1]
        i0 = 2
        for j0 in [0, 2]:
            for jp in range(2):
                for kp in range(2):
                    u = {i0*m + (j0+jp): 1}
                    v = {(j0+jp)*p + (k0+kp): 1}
                    w = {(k0+kp)*n + i0: 1}
                    terms.append((u, v, w))
        
        # 1x1x2 for middle [4]
        i0 = 2
        j0 = 4
        for kp in range(2):
            u = {i0*m + j0: 1}
            v = {j0*p + (k0+kp): 1}
            w = {(k0+kp)*n + i0: 1}
            terms.append((u, v, w))
    
    # Build U, V, W matrices
    rank = len(terms)
    U = [[0]*rank for _ in range(n*m)]
    V = [[0]*rank for _ in range(m*p)]
    W = [[0]*rank for _ in range(p*n)]
    
    for col, (u, v, w) in enumerate(terms):
        for idx, val in u.items():
            U[idx][col] = val
        for idx, val in v.items():
            V[idx][col] = val
        for idx, val in w.items():
            W[idx][col] = val
    
    return {"U": U, "V": V, "W": W}
