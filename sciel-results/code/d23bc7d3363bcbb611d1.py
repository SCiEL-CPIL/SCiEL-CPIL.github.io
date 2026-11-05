def construct():
    n, m, p = 3, 4, 5
    
    n_parts = [1, 2]
    m_parts = [2, 2]
    p_parts = [2, 3]
    
    n_offsets = [0]
    for s in n_parts[:-1]:
        n_offsets.append(n_offsets[-1] + s)
    m_offsets = [0]
    for s in m_parts[:-1]:
        m_offsets.append(m_offsets[-1] + s)
    p_offsets = [0]
    for s in p_parts[:-1]:
        p_offsets.append(p_offsets[-1] + s)
    
    terms = []  # list of (u_vec, v_vec, w_vec) where each is a flat list
    
    def add_subproblem(ni, mj, pk):
        ns = n_parts[ni]
        ms = m_parts[mj]
        ps = p_parts[pk]
        ni_off = n_offsets[ni]
        mj_off = m_offsets[mj]
        pk_off = p_offsets[pk]
        
        if ns == 2 and ms == 2 and ps == 2:
            # Strassen
            strassen = [
                ([1,0,0,1], [1,0,0,1], [1,0,0,-1]),
                ([0,0,1,1], [1,0,0,0], [0,1,0,0]),
                ([1,0,0,0], [0,1,0,-1], [0,0,1,0]),
                ([0,0,0,1], [0,0,1,-1], [1,0,1,0]),
                ([1,1,0,0], [0,0,0,1], [-1,0,1,0]),
                ([1,0,-1,0], [1,1,0,0], [0,0,0,1]),
                ([0,1,0,-1], [0,0,1,1], [1,0,0,1]),
            ]
            for u_loc, v_loc, w_loc in strassen:
                u_full = [0.0] * (n*m)
                v_full = [0.0] * (m*p)
                w_full = [0.0] * (p*n)
                for li in range(ns):
                    for lj in range(ms):
                        u_full[(ni_off+li)*m + (mj_off+lj)] = u_loc[li*ms+lj]
                for lj in range(ms):
                    for lk in range(ps):
                        v_full[(mj_off+lj)*p + (pk_off+lk)] = v_loc[lj*ps+lk]
                for lk in range(ps):
                    for li in range(ns):
                        w_full[(pk_off+lk)*n + (ni_off+li)] = w_loc[lk*ns+li]
                terms.append((u_full, v_full, w_full))
        else:
            for li in range(ns):
                for lj in range(ms):
                    for lk in range(ps):
                        u_full = [0.0] * (n*m)
                        v_full = [0.0] * (m*p)
                        w_full = [0.0] * (p*n)
                        u_full[(ni_off+li)*m + (mj_off+lj)] = 1.0
                        v_full[(mj_off+lj)*p + (pk_off+lk)] = 1.0
                        w_full[(pk_off+lk)*n + (ni_off+li)] = 1.0
                        terms.append((u_full, v_full, w_full))
    
    for ni in range(len(n_parts)):
        for mj in range(len(m_parts)):
            for pk in range(len(p_parts)):
                add_subproblem(ni, mj, pk)
    
    rank = len(terms)
    U = [[0.0]*rank for _ in range(n*m)]
    V = [[0.0]*rank for _ in range(m*p)]
    W = [[0.0]*rank for _ in range(p*n)]
    
    for c in range(rank):
        u_vec, v_vec, w_vec = terms[c]
        for idx in range(n*m):
            U[idx][c] = u_vec[idx]
        for idx in range(m*p):
            V[idx][c] = v_vec[idx]
        for idx in range(p*n):
            W[idx][c] = w_vec[idx]
    
    return {"U": U, "V": V, "W": W}
