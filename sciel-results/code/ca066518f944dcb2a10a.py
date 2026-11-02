<<<<<<< SEARCH
def _optimize_single(xyz0, deadline):
    """Optimize from a single initial config with adaptive penalty."""
    params = xyz0.flatten().copy()
    best_params = params.copy()
    best_score = np.sum(params[2::3])
    
    bounds = []
    for i in range(N):
        bounds.append((0, 1))
        bounds.append((0, 1))
        bounds.append((0, 0.5))
    
    # Phase 1: k-NN based with moderate penalty (fast, broad)
    remaining = deadline - time.time()
    if remaining > 6:
        penalty = 10.0
        try:
            res = minimize(_objective_knn, params, method='L-BFGS-B',
                          bounds=bounds, args=(penalty, 1e-4, 10),
                          options={'maxiter': 600, 'ftol': 1e-14, 'gtol': 1e-12})
            if res.fun < _objective_knn(params, penalty, 1e-4, 10):
                params = res.x
                best_params = params.copy()
                best_score = np.sum(params[2::3])
        except Exception:
            pass
    
    # Phase 2: k-NN with higher penalty (tighten constraints)
    remaining = deadline - time.time()
    if remaining > 5:
        penalty = 50.0
        try:
            res = minimize(_objective_knn, params, method='L-BFGS-B',
                          bounds=bounds, args=(penalty, 1e-4, 10),
                          options={'maxiter': 500, 'ftol': 1e-14, 'gtol': 1e-12})
            if res.fun < _objective_knn(params, penalty, 1e-4, 10):
                params = res.x
                best_params = params.copy()
                best_score = np.sum(params[2::3])
        except Exception:
            pass
    
    # Phase 3: Full pairwise with high penalty
    remaining = deadline - time.time()
    if remaining > 4:
        penalty = 200.0
        try:
            res = minimize(_objective_full, params, method='L-BFGS-B',
                          bounds=bounds, args=(penalty, 1e-5),
                          options={'maxiter': 400, 'ftol': 1e-14, 'gtol': 1e-13})
            if res.fun < _objective_full(params, penalty, 1e-5):
                params = res.x
                best_params = params.copy()
                best_score = np.sum(params[2::3])
        except Exception:
            pass
    
    # Phase 4: Very high penalty, full pairwise
    remaining = deadline - time.time()
    if remaining > 3:
        penalty = 1000.0
        try:
            res = minimize(_objective_full, params, method='L-BFGS-B',
                          bounds=bounds, args=(penalty, 1e-6),
                          options={'maxiter': 300, 'ftol': 1e-15, 'gtol': 1e-13})
            if res.fun < _objective_full(params, penalty, 1e-6):
                params = res.x
                best_params = params.copy()
                best_score = np.sum(params[2::3])
        except Exception:
            pass
    
    # Phase 5: Final high-precision polish
    remaining = deadline - time.time()
    if remaining > 2:
        penalty = 5000.0
        try:
            res = minimize(_objective_full, params, method='L-BFGS-B',
                          bounds=bounds, args=(penalty, 1e-7),
                          options={'maxiter': 250, 'ftol': 1e-15, 'gtol': 1e-14, 'maxls': 30})
            if res.fun < _objective_full(params, penalty, 1e-7):
                params = res.x
                best_params = params.copy()
                best_score = np.sum(params[2::3])
        except Exception:
            pass
    
    # Phase 6: Ultra-precise final polish with smallest smoothness
    remaining = deadline - time.time()
    if remaining > 1.5:
        penalty = 10000.0
        try:
            res = minimize(_objective_full, params, method='L-BFGS-B',
                          bounds=bounds, args=(penalty, 1e-8),
                          options={'maxiter': 200, 'ftol': 1e-16, 'gtol': 1e-15, 'maxls': 40})
            if res.fun < _objective_full(params, penalty, 1e-8):
                params = res.x
                best_params = params.copy()
                best_score = np.sum(params[2::3])
        except Exception:
            pass
    
    return best_params.reshape(N, 3), best_score
=======
def _local_radius_refine(xyz, tol=1e-7, max_iter=200):
    """
    Local refinement: try to increase each radius slightly while maintaining
    non-overlap. This exploits the fact that L-BFGS may leave small gaps.
    """
    xyz = xyz.copy()
    n = xyz.shape[0]
    improved = True
    iters = 0
    while improved and iters < max_iter:
        improved = False
        iters += 1
        for i in range(n):
            # Try to increase radius i by small amount
            best_dr = 0.0
            lo, hi = 0.0, 0.005
            for _ in range(25):
                mid = (lo + hi) / 2.0
                xyz[i, 2] += (mid - best_dr)
                # Check if still valid
                if _validate(xyz, tol=1e-6):
                    best_dr = mid
                    lo = mid
                else:
                    xyz[i, 2] -= (mid - best_dr)
                    hi = mid
            xyz[i, 2] += best_dr
            if best_dr > 1e-8:
                improved = True
            else:
                xyz[i, 2] += best_dr
    return xyz
>>>>>>> REPLACE

