def construct():
    n, m, p = 5, 5, 5
    start_time = time.time()
    deadline = start_time + 110.0
    
    terms = []
    
    def add_term(U_dict, V_dict, W_dict):
        if time.time() > deadline:
            return
        U_dict = {k: v for k, v in U_dict.items() if v != 0}
        V_dict = {k: v for k, v in V_dict.items() if v != 0}
        W_dict = {k: v for k, v in W_dict.items() if v != 0}
        if U_dict and V_dict and W_dict:
            terms.append((U_dict, V_dict, W_dict))
            
    def get_strassen_2x2_terms():
        terms = []
        terms.append(({0: 1, 3: 1}, {0: 1, 3: 1}, {0: 1, 3: 1}))
        terms.append(({2: 1, 3: 1}, {0: 1}, {2: 1, 3: -1}))
        terms.append(({0: 1}, {1: 1, 3: -1}, {1: 1, 3: 1}))
        terms.append(({3: 1}, {2: 1, 0: -1}, {0: 1, 2: 1}))
        terms.append(({0: 1, 1: 1}, {3: 1}, {0: -1, 1: 1}))
        terms.append(({2: 1, 0: -1}, {0: 1, 1: 1}, {3: 1}))
        terms.append(({1: 1, 3: -1}, {2: 1, 3: 1}, {0: 1}))
        return terms

    strassen_terms = get_strassen_2x2_terms()
    
    def kron_u(d1, d2):
        res = {}
        for i1, v1 in d1.items():
            r1 = i1 // 2
            c1 = i1 % 2
            for i2, v2 in d2.items():
                r2 = i2 // 2
                c2 = i2 % 2
                new_r = 2 * r1 + r2
                new_c = 2 * c1 + c2
                new_idx = new_r * 4 + new_c
                val = v1 * v2
                if new_idx in res:
                    res[new_idx] += val
                else:
                    res[new_idx] = val
        return res

    def kron_v(d1, d2):
        res = {}
        for i1, v1 in d1.items():
            c1 = i1 // 2
            k1 = i1 % 2
            for i2, v2 in d2.items():
                c2 = i2 // 2
                k2 = i2 % 2
                new_c = 2 * c1 + c2
                new_k = 2 * k1 + k2
                new_idx = new_c * 4 + new_k
                val = v1 * v2
                if new_idx in res:
                    res[new_idx] += val
                else:
                    res[new_idx] = val
        return res

    def kron_w(d1, d2):
        res = {}
        for i1, v1 in d1.items():
            k1 = i1 // 2
            r1 = i1 % 2
            for i2, v2 in d2.items():
                k2 = i2 // 2
                r2 = i2 % 2
                new_k = 2 * k1 + k2
                new_r = 2 * r1 + r2
                new_idx = new_k * 4 + new_r
                val = v1 * v2
                if new_idx in res:
                    res[new_idx] += val
                else:
                    res[new_idx] = val
        return res

    # Add 4x4x4 block at (0,0,0)
    for t1 in strassen_terms:
        for t2 in strassen_terms:
            U1, V1, W1 = t1
            U2, V2, W2 = t2
            add_term(kron_u(U1, U2), kron_v(V1, V2), kron_w(W1, W2))

    # Add schoolbook for other blocks
    def add_schoolbook_block(r1, r2, c1, c2, k1, k2):
        for ii in range(r1, r2):
            for jj in range(c1, c2):
                for kk in range(k1, k2):
                    u_idx = ii * m + jj
                    v_idx = jj * p + kk
                    w_idx = kk * n + ii
                    add_term({u_idx: 1}, {v_idx: 1}, {w_idx: 1})
                    
    blocks = [(0, 4), (4, 5)]
    for r1, r2 in blocks:
        for c1, c2 in blocks:
            for k1, k2 in blocks:
                if time.time() > deadline:
                    break
                if r1 == 0 and c1 == 0 and k1 == 0 and r2 == 4 and c2 == 4 and k2 == 4:
                    continue # Already added
                add_schoolbook_block(r1, r2, c1, c2, k1, k2)
            if time.time() > deadline:
                break
        if time.time() > deadline:
            break
            
    rank = len(terms)
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for r in range(rank):
        U_dict, V_dict, W_dict = terms[r]
        for idx, val in U_dict.items():
            U[idx][r] = val
        for idx, val in V_dict.items():
            V[idx][r] = val
        for idx, val in W_dict.items():
            W[idx][r] = val
            
    return {"U": U, "V": V, "W": W}
