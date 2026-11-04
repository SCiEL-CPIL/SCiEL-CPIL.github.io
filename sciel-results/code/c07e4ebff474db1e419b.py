def construct():
    start_time = time.time()
    DEADLINE = 140
    
    n, m, p = 4, 4, 5
    nm, mp, pn = n*m, m*p, p*n
    num_eq = n * m * p  # 80
    
    # Precompute the structure
    # For each equation (i,j,k), the coefficient of U[i*m+j, t] is V[j*p+k, t] * W[k*n+i, t]
    # The coefficient of V[j*p+k, t] is U[i*m+j, t] * W[k*n+i, t]
    # The coefficient of W[k*n+i, t] is U[i*m+j, t] * V[j*p+k, t]
    
    def alt_minimize(U, V, W, r, max_iters=100):
        """Alternating minimization to reduce residual."""
        for it in range(max_iters):
            if time.time() - start_time > DEADLINE - 10:
                break
            
            # Update U
            # Build A_U: shape (num_eq, nm * r)
            # A_U[eq_idx, (i*m+j)*r + t] = V[j*p+k, t] * W[k*n+i, t]
            A_U = np.zeros((num_eq, nm * r))
            b = np.ones(num_eq)
            eq_idx = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        for t in range(r):
                            A_U[eq_idx, (i*m+j)*r + t] = V[j*p+k, t] * W[k*n+i, t]
                        eq_idx += 1
            
            sol = np.linalg.lstsq(A_U, b, rcond=None)[0]
            U_new = sol.reshape(nm, r)
            
            # Update V
            A_V = np.zeros((num_eq, mp * r))
            eq_idx = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        for t in range(r):
                            A_V[eq_idx, (j*p+k)*r + t] = U_new[i*m+j, t] * W[k*n+i, t]
                        eq_idx += 1
            sol = np.linalg.lstsq(A_V, b, rcond=None)[0]
            V_new = sol.reshape(mp, r)
            
            # Update W
            A_W = np.zeros((num_eq, pn * r))
            eq_idx = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        for t in range(r):
                            A_W[eq_idx, (k*n+i)*r + t] = U_new[i*m+j, t] * V_new[j*p+k, t]
                        eq_idx += 1
            sol = np.linalg.lstsq(A_W, b, rcond=None)[0]
            W_new = sol.reshape(pn, r)
            
            U, V, W = U_new, V_new, W_new
            
            # Check residual
            max_res = 0.0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        val = sum(U[i*m+j, t] * V[j*p+k, t] * W[k*n+i, t] for t in range(r))
                        max_res = max(max_res, abs(val - 1.0))
            if max_res < 1e-8:
                break
        
        return U, V, W
    
    best_rank = n * m * p
    best_U, best_V, best_W = None, None, None
    
    # Try ranks from 22 down to 15
    for target_rank in range(22, 14, -1):
        if time.time() - start_time > DEADLINE - 20:
            break
        
        for trial in range(2):
            if time.time() - start_time > DEADLINE - 10:
                break
            
            U = np.random.randn(nm, target_rank) * 0.1
            V = np.random.randn(mp, target_rank) * 0.1
            W = np.random.randn(pn, target_rank) * 0.1
            
            U, V, W = alt_minimize(U, V, W, target_rank, max_iters=50)
            
            # Check residual
            max_res = 0.0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        val = sum(U[i*m+j, t] * V[j*p+k, t] * W[k*n+i, t] for t in range(target_rank))
                        max_res = max(max_res, abs(val - 1.0))
            
            if max_res < 1e-6:
                best_rank = target_rank
                best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
                break
        
        if best_U is not None:
            break
    
    if best_U is None:
        rank = n * m * p
        U = np.zeros((nm, rank))
        V = np.zeros((mp, rank))
        W = np.zeros((pn, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j, col] = 1
                    V[j*p+k, col] = 1
                    W[k*n+i, col] = 1
                    col += 1
        best_U, best_V, best_W, best_rank = U, V, W, rank
    
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
