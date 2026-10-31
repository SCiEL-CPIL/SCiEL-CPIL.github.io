<<<<<<< SEARCH
    # Phase configurations: (method, beta, maxiter, ftol, gtol, time_budget)
    # Phase 1: Powell coarse search
    # Phase 2: L-BFGS with low beta (smooth, broad)
    # Phase 3: L-BFGS with medium beta (sharper)
    # Phase 4: L-BFGS with high beta (near-exact)
    phase_configs = [
        ('powell', None, 300, None, None, 15),
        ('lbfgs_soft', 10.0, 500, 1e-8, 1e-5, 15),
        ('lbfgs_soft', 50.0, 1000, 1e-10, 1e-7, 15),
        ('lbfgs_soft', 200.0, 2000, 1e-12, 1e-9, 15),
    ]
    
    best_score_overall = 1e10
    best_params_overall = None
    
    # Process seeds in order of expected quality
    for seed_pts in all_seeds:
        if time_left() < 60:
            break
            
        params, score = optimize_seed(seed_pts, phase_configs)
        
        if params is not None and score < best_score_overall:
            best_score_overall = score
            best_params_overall = params.copy()
    
    # Final polish on the best found
    if best_params_overall is not None:
        best_params_overall, best_score_overall = final_polish(best_params_overall)
=======
    # Phase configurations: (method, beta, maxiter, ftol, gtol, time_budget)
    # Phase 1: SA for broad exploration
    # Phase 2: Powell coarse search
    # Phase 3: L-BFGS with low beta (smooth, broad)
    # Phase 4: L-BFGS with medium beta (sharper)
    # Phase 5: L-BFGS with high beta (near-exact)
    phase_configs = [
        ('sa', None, None, None, None, 8),
        ('powell', None, 200, None, None, 10),
        ('lbfgs_soft', 10.0, 500, 1e-8, 1e-5, 10),
        ('lbfgs_soft', 50.0, 1000, 1e-10, 1e-7, 10),
        ('lbfgs_soft', 200.0, 2000, 1e-12, 1e-9, 10),
    ]
    
    best_score_overall = 1e10
    best_params_overall = None
    
    # Process seeds in order of expected quality
    for seed_pts in all_seeds:
        if time_left() < 45:
            break
            
        params, score = optimize_seed(seed_pts, phase_configs)
        
        if params is not None and score < best_score_overall:
            best_score_overall = score
            best_params_overall = params.copy()
    
    # Final polish on the best found
    if best_params_overall is not None:
        best_params_overall, best_score_overall = final_polish(best_params_overall)
>>>>>>> REPLACE
