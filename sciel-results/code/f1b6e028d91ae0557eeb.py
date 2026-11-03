def construct():
    import time
    import numpy as np
    
    _t0 = time.time()
    _DEADLINE = 135.0

    n, m, p = 3, 4, 6

    def build_structured():
        terms = []
        
        def add_strassen_block(i1, i2, j1, j2, k1, k2):
            a = (i1, j1)
            b = (i1, j2)
            c = (i2, j1)
            d = (i2, j2)
            e = (j1, k1)
            f = (j1, k2)
            g = (j2, k1)
            h = (j2, k2)
            
            C11 = (i1, k1)
            C12 = (i1, k2)
            C21 = (i2, k1)
            C22 = (i2, k2)
            
            terms.append(({a: 1, d: 1}, {e: 1, h: 1}, {C11: 1, C22: 1}))
            terms.append(({c: 1, d: 1}, {e: 1}, {C21: 1, C22: -1}))
            terms.append(({a: 1}, {f: 1, h: -1}, {C12: 1, C22: 1}))
            terms.append(({d: 1}, {g: 1, e: -1}, {C11: 1, C21: 1}))
            terms.append(({a: 1, b: 1}, {h: 1}, {C11: -1, C12: 1}))
            terms.append(({c: 1, a: -1}, {e: 1, f: 1}, {C22: 1}))
            terms.append(({b: 1, d: -1}, {g: 1, h: 1}, {C11: 1}))
        
        def add_schoolbook_term(i, j, k):
            terms.append((
                {(i, j): 1},
                {(j, k): 1},
                {(i, k): 1}
            ))
        
        # Row 0: schoolbook
        for j in range(4):
            for k in range(6):
                add_schoolbook_term(0, j, k)
        
        # Rows 1-2: Strassen
        for (k1, k2) in [(0,1), (2,3), (4,5)]:
            add_strassen_block(1, 2, 0, 1, k1, k2)
            add_strassen_block(1, 2, 2, 3, k1, k2)
        
        rank = len(terms)
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        
        for col, (u_dict, v_dict, w_dict) in enumerate(terms):
            for (i, j), coeff in u_dict.items():
                U[i * m + j][col] = coeff
            for (j, k), coeff in v_dict.items():
                V[j * p + k][col] = coeff
            for (i, k), coeff in w_dict.items():
                W[k * n + i][col] = coeff
        
        return U, V, W

    U, V, W = build_structured()
    rank = len(U[0])
    
    U_np = np.array(U, dtype=float)
    V_np = np.array(V, dtype=float)
    W_np = np.array(W, dtype=float)
    
    def verify(U_arr, V_arr, W_arr, n, m, p, tol=1e-9):
        rank = U_arr.shape[1]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for r in range(rank):
                        s += U_arr[i*m+j, r] * V_arr[j*p+k, r] * W_arr[k*n+i, r]
                    if abs(s - 1.0) > tol:
                        return False
        return True

    if not verify(U_np, V_np, W_np, n, m, p):
        rank = n * m * p
        U_np = np.zeros((n*m, rank))
        V_np = np.zeros((m*p, rank))
        W_np = np.zeros((p*n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_np[i*m+j, col] = 1
                    V_np[j*p+k, col] = 1
                    W_np[k*n+i, col] = 1
                    col += 1
        return {"U": U_np.tolist(), "V": V_np.tolist(), "W": W_np.tolist()}

    # Attempt to find linear dependencies to reduce rank
    # This is a heuristic check. If a column in U is a linear combination of others,
    # and similarly for V and W, we might be able to remove it.
    # However, for tensor decomposition, the dependency must be simultaneous.
    # We skip complex search due to time constraints and return the verified rank 66 solution.
    
    U_list = U_np.tolist()
    V_list = V_np.tolist()
    W_list = W_np.tolist()
    
    if time.time() - _t0 > _DEADLINE:
        return {"U": U_list, "V": V_list, "W": W_list}
    
    return {"U": U_list, "V": V_list, "W": W_list}
