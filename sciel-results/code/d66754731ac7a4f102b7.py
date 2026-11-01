def construct():
    n, m, p = DIMENSIONS  # 2, 2, 5
    deadline = time.time() + 140
    
    def build_rank18():
        # ... existing rank 18 code ...
        pass

    def check_decomposition(U, V, W, rank):
        # ... existing check ...
        pass

    def vectorized_als_search(target_rank, n_iters=1000, n_restarts=20):
        rank = target_rank
        best_err = float('inf')
        best_solution = None
        
        for restart in range(n_restarts):
            if time.time() > deadline - 5:
                break
            
            np.random.seed(restart * 1000 + 42)
            U = np.random.randn(n*m, rank)
            V = np.random.randn(m*p, rank)
            W = np.random.randn(p*n, rank)
            
            # Precompute index mappings for speed
            # U index: a*m + b
            # V index: b*p + c
            # W index: c*n + a
            
            for it in range(n_iters):
                if time.time() > deadline - 5:
                    break
                
                # Update U
                for a in range(n):
                    for b in range(m):
                        i = a * m + b
                        # V rows: b*p + c for c in 0..p-1
                        v_rows = V[b*p : (b+1)*p, :] # shape (p, rank)
                        # W rows: c*n + a for c in 0..p-1
                        w_rows = np.array([W[c*n + a, :] for c in range(p)]) # shape (p, rank)
                        
                        num = np.sum(v_rows * w_rows, axis=0)
                        den = np.sum(v_rows**2 * w_rows**2, axis=0)
                        mask = den > 1e-10
                        U[i, mask] = num[mask] / den[mask]
                        
                # Update V
                for b in range(m):
                    for c in range(p):
                        j = b * p + c
                        # U rows: a*m + b for a in 0..n-1
                        u_rows = np.array([U[a*m + b, :] for a in range(n)]) # shape (n, rank)
                        # W rows: c*n + a for a in 0..n-1
                        w_rows = W[c*n : (c+1)*n, :] # shape (n, rank)
                        
                        num = np.sum(u_rows * w_rows, axis=0)
                        den = np.sum(u_rows**2 * w_rows**2, axis=0)
                        mask = den > 1e-10
                        V[j, mask] = num[mask] / den[mask]
                        
                # Update W
                for c in range(p):
                    for a in range(n):
                        k = c * n + a
                        # U rows: a*m + b for b in 0..m-1
                        u_rows = np.array([U[a*m + b, :] for b in range(m)]) # shape (m, rank)
                        # V rows: b*p + c for b in 0..m-1
                        v_rows = np.array([V[b*p + c, :] for b in range(m)]) # shape (m, rank)
                        
                        num = np.sum(u_rows * v_rows, axis=0)
                        den = np.sum(u_rows**2 * v_rows**2, axis=0)
                        mask = den > 1e-10
                        W[k, mask] = num[mask] / den[mask]
                        
            # Check error
            err = 0.0
            for a in range(n):
                for b in range(m):
                    for c in range(p):
                        i = a*m + b
                        j = b*p + c
                        k = c*n + a
                        val = np.sum(U[i] * V[j] * W[k])
                        err += (val - 1.0)**2
                        
            if err < best_err:
                best_err = err
                best_solution = (U.copy(), V.copy(), W.copy())
                
            if best_err < 1e-6:
                break
                
        if best_solution is not None and best_err < 1e-6:
            return best_solution
        return None

    # Try rank 16
    res16 = vectorized_als_search(16, n_iters=500, n_restarts=10)
    if res16:
        U, V, W = res16
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
        
    # Try rank 17
    res17 = vectorized_als_search(17, n_iters=500, n_restarts=10)
    if res17:
        U, V, W = res17
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
        
    # Fallback to rank 18
    U18, V18, W18 = build_rank18()
    return {"U": U18, "V": V18, "W": W18}
