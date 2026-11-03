<<<<<<< SEARCH
def construct():
    start_time = time.time()
    
    n = N
    outer_center = (0.0, 0.0)
    outer_angle = 0.0
    
    # Initial lattice positions
    positions = _make_lattice_positions(n)
    x0 = np.array([p[0] for p in positions] + [p[1] for p in positions])
    
    # Nelder-Mead optimization
    def objective(params):
        if time.time() - start_time > DEADLINE:
            return 1e8
        return _compute_objective(params, n, outer_center, outer_angle)
    
    best_x = x0.copy()
    best_f = objective(x0)
    
    # Run Nelder-Mead with time check
    try:
        result = minimize(
            objective,
            x0,
            method='Nelder-Mead',
            options={
                'maxiter': 500,
                'maxfev': 2000,
                'xatol': 1e-4,
                'fatol': 1e-4,
                'adaptive': True
            }
        )
        if result.fun < best_f:
            best_x = result.x
            best_f = result.fun
    except Exception:
        pass
    
    # Check deadline
    if time.time() - start_time > DEADLINE - 5:
        # Return best found
        inner = [[best_x[2*i], best_x[2*i+1], 0.0] for i in range(n)]
        return inner, outer_center, best_f, outer_angle
    
    # Try a few more local optimizations from perturbed starts
    rng = np.random.default_rng(42)
    for attempt in range(3):
        if time.time() - start_time > DEADLINE - 10:
            break
        
        x_perturbed = best_x + rng.normal(0, 0.1, size=best_x.shape)
        try:
            result = minimize(
                objective,
                x_perturbed,
                method='Nelder-Mead',
                options={
                    'maxiter': 200,
                    'maxfev': 500,
                    'xatol': 1e-4,
                    'fatol': 1e-4,
                    'adaptive': True
                }
            )
            if result.fun < best_f:
                best_x = result.x
                best_f = result.fun
        except Exception:
            pass
    
    inner = [[best_x[2*i], best_x[2*i+1], 0.0] for i in range(n)]
    return inner, outer_center, best_f, outer_angle
=======
def construct():
    start_time = time.time()
    
    n = N
    outer_center = (0.0, 0.0)
    outer_angle = 0.0
    
    # Initial lattice positions
    positions = _make_lattice_positions(n)
    x0 = np.array([p[0] for p in positions] + [p[1] for p in positions])
    
    # Nelder-Mead optimization
    def objective(params):
        if time.time() - start_time > DEADLINE:
            return 1e8
        return _compute_objective(params, n, outer_center, outer_angle)
    
    best_x = x0.copy()
    best_f = objective(x0)
    
    # Run Nelder-Mead with time check
    try:
        result = minimize(
            objective,
            x0,
            method='Nelder-Mead',
            options={
                'maxiter': 500,
                'maxfev': 2000,
                'xatol': 1e-4,
                'fatol': 1e-4,
                'adaptive': True
            }
        )
        if result.fun < best_f:
            best_x = result.x
            best_f = result.fun
    except Exception:
        pass
    
    # Check deadline
    if time.time() - start_time > DEADLINE - 5:
        # Return best found
        inner = [[best_x[2*i], best_x[2*i+1], 0.0] for i in range(n)]
        return inner, outer_center, best_f, outer_angle
    
    # Try a few more local optimizations from perturbed starts
    rng = np.random.default_rng(42)
    for attempt in range(3):
        if time.time() - start_time > DEADLINE - 10:
            break
        
        x_perturbed = best_x + rng.normal(0, 0.1, size=best_x.shape)
        try:
            result = minimize(
                objective,
                x_perturbed,
                method='Nelder-Mead',
                options={
                    'maxiter': 200,
                    'maxfev': 500,
                    'xatol': 1e-4,
                    'fatol': 1e-4,
                    'adaptive': True
                }
            )
            if result.fun < best_f:
                best_x = result.x
                best_f = result.fun
        except Exception:
            pass
    
    inner = [[best_x[2*i], best_x[2*i+1], 0.0] for i in range(n)]
    return inner, outer_center, best_f, outer_angle
>>>>>>> REPLACE
