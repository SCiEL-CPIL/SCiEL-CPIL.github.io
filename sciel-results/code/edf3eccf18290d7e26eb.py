def _solve_rank(rank, n, m, p, timeout=10):
    start_time = time.time()
    
    num_u = n * m
    num_v = m * p
    num_w = p * n
    total_vars = rank * (num_u + num_v + num_w)
    
    def residuals(x):
        U_flat = x[:rank * num_u]
        V_flat = x[rank * num_u : rank * num_u + rank * num_v]
        W_flat = x[rank * num_u + rank * num_v :]
        
        U = U_flat.reshape(num_u, rank)
        V = V_flat.reshape(num_v, rank)
        W = W_flat.reshape(num_w, rank)
        
        res = []
        for i in range(n):
            for k in range(p):
                for j in range(m):
                    for jp in range(m):
                        s = np.dot(W[k * n + i], U[i * m + j] * V[jp * p + k])
                        target = 1.0 if j == jp else 0.0
                        res.append(s - target)
        return np.array(res)

    for restart in range(50):
        if time.time() - start_time > timeout:
            break
            
        x0 = np.random.randn(total_vars)
        
        try:
            result = opt.least_squares(residuals, x0, method='trf', max_nfev=2000, ftol=1e-15, xtol=1e-15, gtol=1e-15)
        except Exception:
            continue

        if result.cost < 1e-12:
            U_flat = result.x[:rank * num_u]
            V_flat = result.x[rank * num_u : rank * num_u + rank * num_v]
            W_flat = result.x[rank * num_u + rank * num_v :]
            
            U = U_flat.reshape(num_u, rank)
            V = V_flat.reshape(num_v, rank)
            W = W_flat.reshape(num_w, rank)
            
            U_list = U.tolist()
            V_list = V.tolist()
            W_list = W.tolist()
            
            if _verify_solution(U_list, V_list, W_list, n, m, p):
                return U_list, V_list, W_list

    return None
