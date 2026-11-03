<<<<<<< SEARCH
def construct():
    m = 12
    n_roots = 12
    
    seed_roots = [4.0 * (i + 1) for i in range(n_roots)]
    seed_val = _c4_bound(seed_roots, m)
    
    best_roots = seed_roots[:]
    best_val = seed_val
    
    # Phase 0: Try density collocation (new structural approach)
    if _time_left() > 180:
        result = _density_collocation()
        if result is not None:
            roots_candidate, val_candidate = result
            if val_candidate < best_val:
                best_val = val_candidate
                best_roots = roots_candidate
    
    # Phase 1: Optimize in geometric parameter space
    if _time_left() > 120:
        np.random.seed(42)
        
        # Fit geometric to seed
        seed_logs = np.log(seed_roots)
        x = np.arange(n_roots)
        A = np.vstack([np.ones_like(x), x]).T
        try:
            log_r0_fit, log_q_fit = np.linalg.lstsq(A, seed_logs, rcond=None)[0]
        except Exception:
            log_r0_fit, log_q_fit = math.log(4.0), math.log(2.0)
        
        start_points = [
            (log_r0_fit, log_q_fit),
            (math.log(4.0), math.log(2.0)),
            (math.log(2.0), math.log(2.0)),
            (math.log(1.0), math.log(1.5)),
            (math.log(5.0), math.log(1.3)),
            (math.log(3.0), math.log(1.8)),
            (math.log(1.0), math.log(2.0)),
            (math.log(10.0), math.log(1.2)),
            (math.log(0.5), math.log(2.5)),
        ]
        
        for _ in range(8):
            lr0 = np.random.uniform(math.log(0.1), math.log(20.0))
            lq = np.random.uniform(math.log(0.5), math.log(4.0))
            start_points.append((lr0, lq))
        
        for sp in start_points:
            if _time_left() < 60:
                break
            
            try:
                result = minimize(
                    _objective_geometric,
                    x0=np.array(sp),
                    method='L-BFGS-B',
                    bounds=[(math.log(0.01), math.log(100.0)), (math.log(0.1), math.log(10.0))],
                    options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-10}
                )
                
                if result.fun < best_val:
                    roots = _geometric_roots_from_params(result.x)
                    if np.all(roots > 0) and np.all(np.isfinite(roots)):
                        score = _c4_bound(roots, m)
                        if score < best_val:
                            best_val = score
                            best_roots = [float(r) for r in roots]
            except Exception:
                continue
    
    # Phase 2: Direct optimization of all 12 roots in log-space
    if _time_left() > 60:
        def objective_log(log_roots):
            roots = np.exp(log_roots)
            if np.any(roots <= 0):
                return float('inf')
            return _c4_bound(roots, m)
        
        x0 = np.log(np.array(best_roots, dtype=float))
        bounds_low = np.ones(n_roots) * math.log(0.05)
        bounds_high = np.ones(n_roots) * math.log(200.0)
        
        try:
            result = minimize(
                objective_log,
                x0=x0,
                method='L-BFGS-B',
                bounds=list(zip(bounds_low, bounds_high)),
                options={'maxiter': 500, 'ftol': 1e-14, 'gtol': 1e-12}
            )
            
            if np.isfinite(result.fun) and result.fun < best_val:
                roots = np.exp(result.x)
                if np.all(roots > 0) and np.all(np.isfinite(roots)):
                    best_val = result.fun
                    best_roots = [float(r) for r in roots]
        except Exception:
            pass
        
        # Nelder-Mead in log-space
        if _time_left() > 30:
            x0 = np.log(np.array(best_roots, dtype=float))
            best_x, best_f = _nelder_mead(
                objective_log, x0, bounds_low, bounds_high, max_iter=200
            )
            
            if np.isfinite(best_f) and best_f < best_val:
                roots = np.exp(best_x)
                if np.all(roots > 0) and np.all(np.isfinite(roots)):
                    best_val = best_f
                    best_roots = [float(r) for r in roots]
    
    # Phase 3: Fine-tune with perturbations
    if _time_left() > 30 and best_val < seed_val:
        def objective2_log(log_roots):
            roots = np.exp(log_roots)
            if np.any(roots <= 0):
                return float('inf')
            return _c4_bound(roots, m)
        
        x0 = np.log(np.array(best_roots, dtype=float))
        bounds_low = np.ones(n_roots) * math.log(0.02)
        bounds_high = np.ones(n_roots) * math.log(300.0)
        
        for restart in range(5):
            if _time_left() < 20:
                break
            x0_pert = x0 + np.random.normal(0, 0.1, n_roots)
            x0_pert = np.clip(x0_pert, bounds_low, bounds_high)
            
            best_x, best_f = _nelder_mead(
                objective2_log, x0_pert, bounds_low, bounds_high, max_iter=100
            )
            
            if np.isfinite(best_f) and best_f < best_val:
                roots = np.exp(best_x)
                if np.all(roots > 0) and np.all(np.isfinite(roots)):
                    best_val = best_f
                    best_roots = [float(r) for r in roots]
                    x0 = best_x
    
    return best_roots
=======
def construct():
    m = 12
    n_roots = 12
    
    seed_roots = [4.0 * (i + 1) for i in range(n_roots)]
    seed_val = _c4_bound(seed_roots, m)
    
    best_roots = seed_roots[:]
    best_val = seed_val
    
    bounds_low = np.ones(n_roots) * math.log(0.05)
    bounds_high = np.ones(n_roots) * math.log(200.0)
    
    # Phase 1: Differential Evolution global search in log-space
    if _time_left() > 200:
        x0_log = np.log(np.array(best_roots, dtype=float))
        de_best_log, de_best_val = _differential_evolution(
            x0_log, bounds_low, bounds_high, pop_size=24, max_gen=60
        )
        if de_best_val < best_val:
            best_val = de_best_val
            best_roots = [float(r) for r in np.exp(de_best_log)]
    
    # Phase 2: Coordinate descent refinement (fast, gradient-free)
    if _time_left() > 60:
        x_log = np.log(np.array(best_roots, dtype=float))
        x_cd, f_cd = _coordinate_descent(x_log, bounds_low, bounds_high, max_rounds=40)
        if f_cd < best_val:
            best_val = f_cd
            best_roots = [float(r) for r in np.exp(x_cd)]
    
    # Phase 3: Second DE pass with tighter bounds around current best
    if _time_left() > 120:
        x_log = np.log(np.array(best_roots, dtype=float))
        # Tighten bounds around current best
        tl = np.maximum(bounds_low, x_log - 2.0)
        th = np.minimum(bounds_high, x_log + 2.0)
        de_best_log, de_best_val = _differential_evolution(
            x_log, tl, th, pop_size=16, max_gen=40
        )
        if de_best_val < best_val:
            best_val = de_best_val
            best_roots = [float(r) for r in np.exp(de_best_log)]
        
        # Coordinate descent again
        if _time_left() > 30:
            x_log = np.log(np.array(best_roots, dtype=float))
            x_cd, f_cd = _coordinate_descent(x_log, tl, th, max_rounds=30)
            if f_cd < best_val:
                best_val = f_cd
                best_roots = [float(r) for r in np.exp(x_cd)]
    
    # Phase 4: Final fine-tune with small perturbations + coordinate descent
    if _time_left() > 30:
        x_log = np.log(np.array(best_roots, dtype=float))
        np.random.seed(999)
        for _ in range(4):
            if _time_left() < 15:
                break
            x_pert = x_log + np.random.normal(0, 0.05, n_roots)
            x_pert = np.clip(x_pert, bounds_low, bounds_high)
            x_cd, f_cd = _coordinate_descent(x_pert, bounds_low, bounds_high, max_rounds=20)
            if f_cd < best_val:
                best_val = f_cd
                best_roots = [float(r) for r in np.exp(x_cd)]
                x_log = x_cd
    
    return best_roots
>>>>>>> REPLACE
