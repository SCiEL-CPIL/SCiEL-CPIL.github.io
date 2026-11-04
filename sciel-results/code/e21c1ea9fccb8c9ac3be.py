<<<<<<< SEARCH
    # Add points from other alphas
    alphas_to_try = [best_alpha * 0.1, best_alpha * 0.5, best_alpha * 2.0, best_alpha * 10.0, 1.0, 0.1, 10.0]
    for a in alphas_to_try:
        if not time_ok():
            break
        try:
            if a > 0:
                xg_try, _ = roots_genlaguerre(12, a)
                xg_try = np.sort(xg_try)
                if np.all(xg_try > 0) and np.all(np.isfinite(xg_try)):
                    initial_points.append(xg_try)
        except Exception:
            pass

    # Add perturbed versions of the best root
    rng = np.random.default_rng(42)
    for _ in range(5):
        if not time_ok():
            break
        pert = best_roots * (1.0 + 0.3 * rng.standard_normal(12))
        pert = np.maximum(pert, 1e-3)
        pert = np.sort(pert)
        initial_points.append(pert)
=======
    # Add points from other alphas - expanded set
    alphas_to_try = [best_alpha * 0.05, best_alpha * 0.1, best_alpha * 0.3, best_alpha * 0.5,
                     best_alpha * 1.5, best_alpha * 2.0, best_alpha * 5.0, best_alpha * 10.0,
                     0.01, 0.1, 1.0, 10.0, 100.0, 0.5, 2.0]
    for a in alphas_to_try:
        if not time_ok():
            break
        try:
            if a > 0:
                xg_try, _ = roots_genlaguerre(12, a)
                xg_try = np.sort(xg_try)
                if np.all(xg_try > 0) and np.all(np.isfinite(xg_try)):
                    initial_points.append(xg_try)
        except Exception:
            pass

    # Add perturbed versions of the best root with varied scales
    rng = np.random.default_rng(42)
    for scale in [0.1, 0.2, 0.3, 0.5]:
        if not time_ok():
            break
        for _ in range(3):
            if not time_ok():
                break
            pert = best_roots * (1.0 + scale * rng.standard_normal(12))
            pert = np.maximum(pert, 1e-3)
            pert = np.sort(pert)
            initial_points.append(pert)
    
    # Add random positive initializations
    for _ in range(3):
        if not time_ok():
            break
        rand_roots = np.sort(rng.uniform(0.1, 50.0, 12))
        initial_points.append(rand_roots)
>>>>>>> REPLACE

