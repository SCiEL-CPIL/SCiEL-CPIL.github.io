<<<<<<< SEARCH
    # Run Differential Evolution with diverse restarts
    seeds = [42, 123, 456, 789, 101, 202, 303, 404, 505, 606, 707, 808]
    
    for seed in seeds:
        if not time_ok():
            break
        
        try:
            result = differential_evolution(
                cost_func, 
                bounds, 
                seed=seed, 
                maxiter=200, 
                popsize=20, 
                tol=1e-10, 
                polish=False,
                init='sobol'
            )
            
            if result.fun < best_val:
                best_val = result.fun
                best_log_roots = result.x.copy()
                best_roots = np.exp(best_log_roots)
                
        except Exception:
            pass
=======
    # Run Differential Evolution with diverse restarts
    # Reduce number of seeds to save time for polishing, as DE is slow
    seeds = [42, 123, 456]
    
    for seed in seeds:
        if not time_ok():
            break
        
        try:
            result = differential_evolution(
                cost_func, 
                bounds, 
                seed=seed, 
                maxiter=100, 
                popsize=15, 
                tol=1e-8, 
                polish=False,
                init='sobol'
            )
            
            if result.fun < best_val:
                best_val = result.fun
                best_log_roots = result.x.copy()
                best_roots = np.exp(best_log_roots)
                
        except Exception:
            pass
            
    # Also try the high-bound seed directly with minimize
    if time_ok():
        try:
            result_high = minimize(
                cost_with_grad,
                seed_log_roots_high,
                jac=True,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 2000, 'ftol': 1e-15, 'gtol': 1e-12}
            )
            if result_high.fun < best_val:
                best_val = result_high.fun
                best_log_roots = result_high.x.copy()
                best_roots = np.exp(best_log_roots)
        except Exception:
            pass
>>>>>>> REPLACE
