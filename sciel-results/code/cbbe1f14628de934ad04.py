def construct():
    n, m, p = DIMENSIONS  # 3, 5, 6
    
    # Block splits
    n_splits = [1, 2]
    m_splits = [2, 2, 1]
    p_splits = [2, 2, 2]
    
    U_blocks = []
    V_blocks = []
    W_blocks = []
    
    for ni, nb in enumerate(n_splits):
        for mi, mb in enumerate(m_splits):
            for pi, pb in enumerate(p_splits):
                if nb == 2 and mb == 2 and pb == 2:
                    ub, vb, wb = _strassen_2x2_2x2()
                else:
                    ub, vb, wb = _schoolbook_block(nb, mb, pb)
                U_blocks.append((ni, nb, ub))
                V_blocks.append((mi, mb, vb))
                W_blocks.append((pi, pb, wb))
    
    # Now assemble full matrices
    # U: rows indexed by i*m+j for i in [0,n), j in [0,m)
    # V: rows indexed by j*p+k for j in [0,m), k in [0,p)
    # W: rows indexed by k*n+i for k in [0,p), i in [0,n)
    
    total_rank = sum(len(ub[0]) for _, _, ub in U_blocks)
    
    U = [[0]*total_rank for _ in range(n*m)]
    V = [[0]*total_rank for _ in range(m*p)]
    W = [[0]*total_rank for _ in range(p*n)]
    
    col_offset = 0
    for (ni, nb, ub), (mi, mb, vb), (pi, pb, wb) in zip(U_blocks, V_blocks, W_blocks):
        r = len(ub[0])
        # Compute row offsets
        n_off = sum(n_splits[:ni])
        m_off = sum(m_splits[:mi])
        p_off = sum(p_splits[:pi])
        
        for ri in range(nb):
            for rj in range(mb):
                for c in range(r):
                    U[(n_off+ri)*m + (m_off+rj)][col_offset+c] = ub[ri*mb+rj][c]
        
        for rj in range(mb):
            for rk in range(pb):
                for c in range(r):
                    V[(m_off+rj)*p + (p_off+rk)][col_offset+c] = vb[rj*pb+rk][c]
        
        for rk in range(pb):
            for ri in range(nb):
                for c in range(r):
                    W[(p_off+rk)*n + (n_off+ri)][col_offset+c] = wb[rk*nb+ri][c]
        
        col_offset += r
    
    return {"U": U, "V": V, "W": W}
