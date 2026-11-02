<<<<<<< SEARCH
def construct():
    start_time = time.time()
    deadline = start_time + DEADLINE

    lo, hi = _bounds()
    bounds_list = list(zip(lo, hi))

    best_params = None
    best_score = -np.inf

    # Multi-start with different initializations
    initializers = [
        lambda s: _voronoi_initial(s),
        lambda s: _hex_initial(s),
        lambda s: _grid_initial(s),
    ]

    # Use more seeds to explore the landscape better
    seeds = [42, 123, 7, 999, 555, 2024, 31415, 271828, 1234, 5678, 
             11, 22, 33, 44, 55, 66, 77, 88, 99, 100]
    
    for seed in seeds:
        if time.time() > deadline - 5:
            break
        for init_fn in initializers:
            if time.time() > deadline - 5:
                break
            params = init_fn(seed)
            rng = np.random.RandomState(seed + 1000)

            # Multi-penalty L-BFGS-B
            # Use a schedule of penalties to guide the optimizer
            for pen in [100.0, 500.0, 2000.0, 5000.0]:
                if time.time() > deadline - 3:
                    break

                def obj(p, pen=pen):
                    return _penalty_objective(p, penalty=pen)

                def grad(p, pen=pen):
                    return _penalty_grad(p, penalty=pen)

                try:
                    res = minimize(obj, params, jac=grad, method='L-BFGS-B',
                                   bounds=bounds_list,
                                   options={'maxiter': 200, 'maxfun': 300,
                                            'ftol': 1e-12, 'gtol': 1e-8})
                    params = res.x
                except Exception:
                    pass

            # Final polish with high penalty
            if time.time() < deadline - 1:
                try:
                    res = minimize(lambda p: _penalty_objective(p, penalty=10000.0),
                                   params,
                                   jac=lambda p: _penalty_grad(p, penalty=10000.0),
                                   method='L-BFGS-B', bounds=bounds_list,
                                   options={'maxiter': 100, 'maxfun': 150,
                                            'ftol': 1e-12, 'gtol': 1e-8})
                    params = res.x
                except Exception:
                    pass

            # Enforce validity and score
            valid_p = _enforce_validity(params)
            score = _score_valid(valid_p)

            if score > best_score:
                best_score = score
                best_params = valid_p.copy()

    if best_params is None:
        # Fallback: grid
        side = int(np.ceil(np.sqrt(N)))
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-4)
                 for i in range(side) for j in range(side)]
        return np.array(cells[:N])

    return best_params
=======
def construct():
    start_time = time.time()
    deadline = start_time + DEADLINE

    lo, hi = _bounds()
    bounds_list = list(zip(lo, hi))

    best_params = None
    best_score = -np.inf

    # Multi-start with different initializations
    initializers = [
        lambda s: _voronoi_initial(s),
        lambda s: _hex_initial(s),
        lambda s: _grid_initial(s),
    ]

    # Use more seeds to explore the landscape better
    seeds = [42, 123, 7, 999, 555, 2024, 31415, 271828, 1234, 5678, 
             11, 22, 33, 44, 55, 66, 77, 88, 99, 100]
    
    # Track warm-start candidate
    warm_start_params = None
    
    for seed_idx, seed in enumerate(seeds):
        if time.time() > deadline - 5:
            break
        for init_idx, init_fn in enumerate(initializers):
            if time.time() > deadline - 5:
                break
            
            # Use warm start from previous best if available and we're not the first run
            if warm_start_params is not None and seed_idx > 0 and init_idx == 0:
                # Perturb the warm start slightly
                rng_perturb = np.random.RandomState(seed + 5000)
                params = warm_start_params + rng_perturb.uniform(-0.01, 0.01, size=warm_start_params.size)
                # Re-clamp to bounds
                params = np.clip(params, lo, hi)
            else:
                params = init_fn(seed)
            
            rng = np.random.RandomState(seed + 1000)

            # Multi-penalty L-BFGS-B with decreasing soft_tau for better convergence
            # Use a schedule of penalties and soft_tau values
            penalty_schedule = [100.0, 500.0, 2000.0, 5000.0, 10000.0]
            tau_schedule = [0.05, 0.02, 0.01, 0.005, 0.002]
            
            for pen_idx, (pen, tau) in enumerate(zip(penalty_schedule, tau_schedule)):
                if time.time() > deadline - 3:
                    break

                def obj(p, pen=pen, tau=tau):
                    return _penalty_objective(p, penalty=pen, soft_tau=tau)

                def grad(p, pen=pen, tau=tau):
                    return _penalty_grad(p, penalty=pen, soft_tau=tau)

                try:
                    res = minimize(obj, params, jac=grad, method='L-BFGS-B',
                                   bounds=bounds_list,
                                   options={'maxiter': 300, 'maxfun': 400,
                                            'ftol': 1e-14, 'gtol': 1e-10})
                    params = res.x
                except Exception:
                    pass

            # Final high-precision polish with very tight tolerances
            if time.time() < deadline - 1:
                try:
                    def obj_polish(p):
                        return _penalty_objective(p, penalty=20000.0, soft_tau=0.001)

                    def grad_polish(p):
                        return _penalty_grad(p, penalty=20000.0, soft_tau=0.001)

                    res = minimize(obj_polish, params, jac=grad_polish,
                                   method='L-BFGS-B', bounds=bounds_list,
                                   options={'maxiter': 500, 'maxfun': 600,
                                            'ftol': 1e-16, 'gtol': 1e-12})
                    params = res.x
                except Exception:
                    pass

            # Enforce validity and score
            valid_p = _enforce_validity(params)
            score = _score_valid(valid_p)

            if score > best_score:
                best_score = score
                best_params = valid_p.copy()
                # Update warm start for subsequent runs
                warm_start_params = valid_p.copy()

    if best_params is None:
        # Fallback: grid
        side = int(np.ceil(np.sqrt(N)))
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-4)
                 for i in range(side) for j in range(side)]
        return np.array(cells[:N])

    return best_params
>>>>>>> REPLACE
