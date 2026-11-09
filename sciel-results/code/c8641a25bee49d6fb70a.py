# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    nm, mp, pn = n*m, m*p, p*n
    
    # Schoolbook decomposition
    rank_school = n * m * p
    U = np.zeros((nm, rank_school))
    V = np.zeros((mp, rank_school))
    W = np.zeros((pn, rank_school))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m + j, col] = 1
                V[j*p + k, col] = 1
                W[k*n + i, col] = 1
                col += 1
    
    best_U, best_V, best_W = U, V, W
    best_rank = rank_school
    
    start_time = time.time()
    deadline = start_time + 110
    
    # Build support indices
    support = []
    sup_i = []
    sup_j = []
    sup_k = []
    for a in range(n):
        for b in range(m):
            for c in range(p):
                i_idx = a*m + b
                j_idx = b*p + c
                k_idx = c*n + a
                support.append((i_idx, j_idx, k_idx))
                sup_i.append(i_idx)
                sup_j.append(j_idx)
                sup_k.append(k_idx)
                
    sup_i = np.array(sup_i)
    sup_j = np.array(sup_j)
    sup_k = np.array(sup_k)
    
    # Pre-sample off-support indices
    # We need to check if (i, j, k) is off-support.
    # i = a*m+b, j = b'*p+c, k = c'*n+a'
    # Off-support if b != b' or c != c' or a != a'
    
    # Generate a fixed set of off-support indices for optimization
    # To ensure we cover the space, we pick random ones.
    np.random.seed(123)
    off_i = []
    off_j = []
    off_k = []
    count = 0
    while count < 200:
        ii = np.random.randint(0, nm)
        jj = np.random.randint(0, mp)
        kk = np.random.randint(0, pn)
        a, b = divmod(ii, m)
        b2, c = divmod(jj, p)
        c2, a2 = divmod(kk, n)
        if b != b2 or c != c2 or a != a2:
            off_i.append(ii)
            off_j.append(jj)
            off_k.append(kk)
            count += 1
            
    off_i = np.array(off_i)
    off_j = np.array(off_j)
    off_k = np.array(off_k)

    for r in range(rank_school - 1, max(80, rank_school - 5), -1):
        if time.time() > deadline:
            break
        
        found = False
        for attempt in range(10):
            if time.time() > deadline:
                break
            
            U_try = np.random.randn(nm, r) * 0.5
            V_try = np.random.randn(mp, r) * 0.5
            W_try = np.random.randn(pn, r) * 0.5
            
            lr = 0.01
            for step in range(100):
                if time.time() > deadline:
                    break
                
                # Compute T_sup
                U_sup = U_try[sup_i]
                V_sup = V_try[sup_j]
                W_sup = W_try[sup_k]
                T_sup = np.sum(U_sup * V_sup * W_sup, axis=1)
                err_sup = T_sup - 1.0
                
                # Compute T_off
                U_off = U_try[off_i]
                V_off = V_try[off_j]
                W_off = W_try[off_k]
                T_off = np.sum(U_off * V_off * W_off, axis=1)
                
                # Gradients
                # dL/dU[i,r] = -2 * sum_{j,k in sup, i} err_sup[idx] * V[j,r] W[k,r] 
                #             + 2 * sum_{j,k in off, i} T_off[idx] * V[j,r] W[k,r]
                
                grad_U = np.zeros_like(U_try)
                grad_V = np.zeros_like(V_try)
                grad_W = np.zeros_like(W_try)
                
                # Vectorized gradient calculation is tricky with indices.
                # We can use np.add.at or similar, but it's slow.
                # Given the small size (nm=15, mp=30, pn=18, r~85),
                # a loop over support (90) and off (200) is acceptable.
                
                for idx in range(len(sup_i)):
                    i_idx = sup_i[idx]
                    j_idx = sup_j[idx]
                    k_idx = sup_k[idx]
                    e = err_sup[idx]
                    v_row = V_try[j_idx]
                    w_row = W_try[k_idx]
                    u_row = U_try[i_idx]
                    
                    grad_U[i_idx] -= 2.0 * e * (v_row * w_row)
                    grad_V[j_idx] -= 2.0 * e * (u_row * w_row)
                    grad_W[k_idx] -= 2.0 * e * (u_row * v_row)
                    
                for idx in range(len(off_i)):
                    i_idx = off_i[idx]
                    j_idx = off_j[idx]
                    k_idx = off_k[idx]
                    t = T_off[idx]
                    v_row = V_try[j_idx]
                    w_row = W_try[k_idx]
                    u_row = U_try[i_idx]
                    
                    grad_U[i_idx] += 2.0 * t * (v_row * w_row)
                    grad_V[j_idx] += 2.0 * t * (u_row * w_row)
                    grad_W[k_idx] += 2.0 * t * (u_row * v_row)
                
                U_try -= lr * grad_U
                V_try -= lr * grad_V
                W_try -= lr * grad_W
                
                # Check convergence
                if step % 10 == 0:
                    max_err = np.max(np.abs(T_sup - 1.0))
                    max_off_err = np.max(np.abs(T_off))
                    if max_err < 1e-6 and max_off_err < 1e-6:
                        # Verify with more points
                        valid = True
                        for _ in range(100):
                            ii = np.random.randint(0, nm)
                            jj = np.random.randint(0, mp)
                            kk = np.random.randint(0, pn)
                            a, b = divmod(ii, m)
                            b2, c = divmod(jj, p)
                            c2, a2 = divmod(kk, n)
                            if b != b2 or c != c2 or a != a2:
                                t_val = np.dot(U_try[ii] * V_try[jj], W_try[kk])
                                if abs(t_val) > 1e-5:
                                    valid = False
                                    break
                        if valid:
                            best_U, best_V, best_W = U_try, V_try, W_try
                            best_rank = r
                            found = True
                            break
                
            if found:
                break
        
        if found:
            break
            
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
# EVOLVE-BLOCK-END
