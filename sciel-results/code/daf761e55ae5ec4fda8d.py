def construct():
    import time
    import numpy as np
    start_time = time.time()
    deadline = start_time + 110  # safety margin before 120s limit
    
    n, m, p = 4, 5, 5
    
    # First, try to find a valid Winograd 2x2x2 scheme
    winograd_scheme = None
    
    # Try several candidate schemes
    candidate_schemes = [
        # Scheme A
        (
            np.array([[1,1,0,0],[0,0,1,1],[1,-1,0,0],[0,0,1,-1]], dtype=float),
            np.array([[1,1,0,0],[0,0,1,1],[1,-1,0,0],[0,0,1,-1]], dtype=float),
            np.array([[1,1,0,0],[1,-1,0,0],[0,1,1,0],[0,-1,1,0]], dtype=float)
        ),
        # Scheme B - Strassen-like
        (
            np.array([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,1,1]], dtype=float),
            np.array([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,1,1]], dtype=float),
            np.array([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,1,1]], dtype=float)
        ),
    ]
    
    for U_cand, V_cand, W_cand in candidate_schemes:
        # Verify: sum_t U[i,t]*V[j,t]*W[k,t] == M[i,j,k]
        ok = True
        for i in range(4):
            for j in range(4):
                for k in range(4):
                    s = np.dot(U_cand[i], V_cand[j] * W_cand[k])
                    r, mm = i//2, i%2
                    mp, c = j//2, j%2
                    rp, cp = k//2, k%2
                    expected = 1.0 if (r==rp and mm==mp and c==cp) else 0.0
                    if abs(s - expected) > 1e-9:
                        ok = False
                        break
                if not ok:
                    break
            if not ok:
                break
        if ok:
            winograd_scheme = (U_cand, V_cand, W_cand)
            break
    
    # Build the decomposition
    terms_U = []
    terms_V = []
    terms_W = []
    
    def add_term(u_entries, v_entries, w_entries):
        terms_U.append(u_entries)
        terms_V.append(v_entries)
        terms_W.append(w_entries)
    
    row_blocks = [(0, 2), (2, 4)]
    mid_blocks = [(0, 2), (2, 4), (4, 5)]
    col_blocks = [(0, 2), (2, 4), (4, 5)]
    
    for (r0, r1) in row_blocks:
        for (m0, m1) in mid_blocks:
            for (k0, k1) in col_blocks:
                rs = r1 - r0
                ms = m1 - m0
                ks = k1 - k0
                
                if time.time() > deadline:
                    break
                
                if rs == 2 and ms == 2 and ks == 2 and winograd_scheme is not None:
                    U_w, V_w, W_w = winograd_scheme
                    for t in range(7):
                        u_entries = []
                        v_entries = []
                        w_entries = []
                        for dr in range(2):
                            for dm in range(2):
                                u_idx = (r0 + dr) * m + (m0 + dm)
                                u_entries.append((u_idx, U_w[dr*2+dm, t]))
                        for dm in range(2):
                            for dk in range(2):
                                v_idx = (m0 + dm) * p + (k0 + dk)
                                v_entries.append((v_idx, V_w[dm*2+dk, t]))
                        for dr in range(2):
                            for dk in range(2):
                                w_idx = (k0 + dk) * n + (r0 + dr)
                                w_entries.append((w_idx, W_w[dk*2+dr, t]))
                        add_term(u_entries, v_entries, w_entries)
                else:
                    # Schoolbook
                    for dr in range(rs):
                        for dm in range(ms):
                            for dk in range(ks):
                                u_idx = (r0 + dr) * m + (m0 + dm)
                                v_idx = (m0 + dm) * p + (k0 + dk)
                                w_idx = (k0 + dk) * n + (r0 + dr)
                                add_term([(u_idx, 1)], [(v_idx, 1)], [(w_idx, 1)])
    
    rank = len(terms_U)
    
    # Build dense matrices
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    for t in range(rank):
        for idx, val in terms_U[t]:
            U[idx, t] = val
        for idx, val in terms_V[t]:
            V[idx, t] = val
        for idx, val in terms_W[t]:
            W[idx, t] = val
    
    # Verify correctness
    if time.time() < deadline - 5:
        np.random.seed(42)
        A_test = np.random.randn(n, m)
        B_test = np.random.randn(m, p)
        C_expected = A_test @ B_test
        C_computed = np.zeros((n, p))
        for t in range(rank):
            u_vec = np.zeros(n * m)
            v_vec = np.zeros(m * p)
            w_vec = np.zeros(p * n)
            for idx, val in terms_U[t]:
                u_vec[idx] = val
            for idx, val in terms_V[t]:
                v_vec[idx] = val
            for idx, val in terms_W[t]:
                w_vec[idx] = val
            prod = np.dot(u_vec, A_test.flatten()) * np.dot(v_vec, B_test.flatten())
            C_computed += prod * w_vec.reshape(p, n).T
        if not np.allclose(C_computed, C_expected, atol=1e-9):
            # Fallback to schoolbook
            terms_U.clear()
            terms_V.clear()
            terms_W.clear()
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        add_term([(i*m+j, 1)], [(j*p+k, 1)], [(k*n+i, 1)])
            rank = len(terms_U)
            U = np.zeros((n * m, rank))
            V = np.zeros((m * p, rank))
            W = np.zeros((p * n, rank))
            for t in range(rank):
                for idx, val in terms_U[t]:
                    U[idx, t] = val
                for idx, val in terms_V[t]:
                    V[idx, t] = val
                for idx, val in terms_W[t]:
                    W[idx, t] = val
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
