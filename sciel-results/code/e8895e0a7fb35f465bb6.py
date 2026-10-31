def _local_optimize(params, maxiter_powell=200, maxiter_lbfgs=300):
    """
    Two-phase local optimization:
    1. Powell on exact objective
    2. L-BFGS-B on soft objective with annealing
    Returns best params and exact score.
    """
    best_params = params.copy()
    best_score = _exact_objective(params)
    
    # Phase 1: Powell
    try:
        res_powell = minimize(
            _exact_objective,
            best_params,
            method='Powell',
            options={'maxiter': maxiter_powell, 'xtol': 1e-8, 'ftol': 1e-8, 'disp': False}
        )
        if res_powell.fun < best_score:
            best_score = res_powell.fun
            best_params = res_powell.x.copy()
    except Exception:
        pass
    
    # Phase 2: L-BFGS-B with annealing on soft objective
    betas = [10.0, 50.0, 200.0, 1000.0]
    for beta in betas:
        try:
            res_lbfgs = minimize(
                lambda p: _soft_objective(p, beta),
                best_params,
                method='L-BFGS-B',
                options={'maxiter': maxiter_lbfgs, 'ftol': 1e-12, 'gtol': 1e-8, 'disp': False}
            )
            exact_s = _exact_objective(res_lbfgs.x)
            if exact_s < best_score:
                best_score = exact_s
                best_params = res_lbfgs.x.copy()
        except Exception:
            pass
    
    return best_params, best_score
