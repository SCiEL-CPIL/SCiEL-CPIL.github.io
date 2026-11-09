<<<<<<< SEARCH
    de_configs = [
        {'seed': 123, 'maxiter': 300, 'popsize': 25},
        {'seed': 456, 'maxiter': 200, 'popsize': 20}
    ]

    for cfg in de_configs:
        if time.time() > deadline - 60:
            break
        try:
            result = differential_evolution(
                cost_func, 
                bounds, 
                seed=cfg['seed'], 
                maxiter=cfg['maxiter'], 
                popsize=cfg['popsize'], 
                tol=1e-10, 
                polish=True,  # L-BFGS-B polish
                init='sobol' # Better initial distribution
            )
            
            if result.fun < best_val:
                best_val = result.fun
                best_log_roots = result.x
                best_roots = np.exp(best_log_roots)
                
        except Exception:
            pass
=======
    # More aggressive DE search
    de_configs = [
        {'seed': 123, 'maxiter': 500, 'popsize': 30},
        {'seed': 456, 'maxiter': 300, 'popsize': 25}
    ]

    for cfg in de_configs:
        if time.time() > deadline - 120:
            break
        try:
            result = differential_evolution(
                cost_func, 
                bounds, 
                seed=cfg['seed'], 
                maxiter=cfg['maxiter'], 
                popsize=cfg['popsize'], 
                tol=1e-12, 
                polish=True,  # L-BFGS-B polish
                init='sobol' # Better initial distribution
            )
            
            if result.fun < best_val:
                best_val = result.fun
                best_log_roots = result.x
                best_roots = np.exp(best_log_roots)
                
        except Exception:
            pass
    
    # Basin Hopping to escape local minima
    if time.time() < deadline - 20 and best_log_roots is not None:
        current_log_roots = best_log_roots.copy()
        current_val = best_val
        np.random.seed(12345) # Fixed seed for reproducibility in this phase
        
        for i in range(100):
            if time.time() > deadline - 10:
                break
            
            # Annealing step size
            step_size = 0.5 * (1.0 - i/100.0)
            if step_size < 0.05:
                step_size = 0.05
                
            perturbed_log_roots = current_log_roots + np.random.normal(0, step_size, 12)
            
            try:
                res = minimize(
                    cost_func, perturbed_log_roots, method='L-BFGS-B',
                    bounds=bounds, options={'maxiter': 200, 'ftol': 1e-10}
                )
                
                if res.fun < current_val:
                    current_val = res.fun
                    current_log_roots = res.x
                    
                    if current_val < best_val:
                        best_val = current_val
                        best_log_roots = current_log_roots.copy()
                        best_roots = np.exp(best_log_roots)
                        
            except Exception:
                pass
>>>>>>> REPLACE
