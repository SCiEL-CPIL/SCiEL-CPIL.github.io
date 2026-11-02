def _try_rank(target_rank, n=2, m=3, p=3, max_trials=100, time_budget=30, seed=42):
    np.random.seed(seed)
    start_time = time.time()
    
    for trial in range(max_trials):
        if time.time() - start_time > time_budget:
            break
        
        # Random initialization with small integers, including 0
        U = np.random.choice([-2, -1, 0, 1, 2], size=(n*m, target_rank)).astype(float)
        V = np.random.choice([-2, -1, 0, 1, 2], size=(m*p, target_rank)).astype(float)
        W = np.random.choice([-2, -1, 0, 1, 2], size=(p*n, target_rank)).astype(float)
        
        # Run ALS
        U_opt, V_opt, W_opt = _als_optimize(U, V, W, n, m, p, max_iter=80)
        
        # Check if exact and close to half-integers
        if _verify(U_opt.tolist(), V_opt.tolist(), W_opt.tolist(), n, m, p, tol=1e-8):
            U_rounded = _round_to_half_integer(U_opt, tol=1e-5)
            V_rounded = _round_to_half_integer(V_opt, tol=1e-5)
            W_rounded = _round_to_half_integer(W_opt, tol=1e-5)
            
            if U_rounded is not None and V_rounded is not None and W_rounded is not None:
                if _verify(U_rounded.tolist(), V_rounded.tolist(), W_rounded.tolist(), n, m, p, tol=1e-10):
                    return {"U": U_rounded.tolist(), "V": V_rounded.tolist(), "W": W_rounded.tolist()}
                    
    return None
