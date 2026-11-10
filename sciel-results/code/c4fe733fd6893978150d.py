<<<<<<< SEARCH
def heilbronn_convex13() -> np.ndarray:
    """
    Construct 13 points to maximize A_min / A_hull.
    Uses known hexagonal structure as the primary seed, then refines.
    """
    n = 13
    start_time = time.time()
    deadline = start_time + 580  # Safety margin
    
    best_points = None
    best_score = 0.0
    
    # Primary seed: known good hexagonal configuration
    # 1 center + 6 on inner hexagon + 6 on outer hexagon
    # For 13 points, the optimal ratio is around 0.734
    # Try several radius ratios
    
    configs = []
    
    # Hexagonal: center + 6 inner + 6 outer
    for r1 in [0.15, 0.18, 0.20, 0.22]:
        for r2 in [0.35, 0.40, 0.42, 0.45]:
            cfg = _build_hex_configuration(r1, r2, (0.5, 0.5))
            configs.append(cfg)
    
    # Concentric pentagon + heptagon
    configs.append(_build_concentric_config(13, (0.5, 0.5)))
    
    # Grid
    configs.append(_build_grid_config(13))
    
    # Random seeds
    for seed in range(10):
        configs.append(_build_random_config(13, seed))
    
    # Evaluate all configs
    scored = []
    for cfg in configs:
        if time.time() > deadline - 300:
            break
        s = _compute_score(cfg)
        scored.append((s, cfg))
    
    scored.sort(key=lambda x: -x[0])
    
    # Take top 5 for refinement
    top_configs = [cfg for _, cfg in scored[:5]]
    
    # Phase 1: Simulated annealing on top configs
    sa_deadline = deadline - 120
    for cfg in top_configs:
        if time.time() > sa_deadline:
            break
        sa_result = _simulated_annealing(cfg, sa_deadline,
                                          T_init=0.025, T_min=1e-9,
                                          cooling_rate=0.9992, max_iters=30000)
        s = _compute_score(sa_result)
        if s > best_score:
            best_score = s
            best_points = sa_result.copy()
    
    # Phase 2: Local search refinement
    if best_points is not None:
        ls_deadline = deadline - 40
        ls_result = _local_refine(best_points, ls_deadline, max_iters=50000)
        s = _compute_score(ls_result)
        if s > best_score:
            best_score = s
            best_points = ls_result.copy()
    
    # Phase 3: Additional random restarts if time permits
    if best_points is not None and time.time() < deadline - 60:
        for _ in range(8):
            if time.time() > deadline - 40:
                break
            perturbed = best_points.copy()
            noise = np.random.randn(*perturbed.shape) * 0.05
            perturbed += noise
            np.clip(perturbed, 0, 1, out=perturbed)
            
            quick_ls = _local_refine(perturbed, deadline - 40, max_iters=15000)
            s = _compute_score(quick_ls)
            if s > best_score:
                best_score = s
                best_points = quick_ls.copy()
    
    # Safety
    if best_points is None or not np.all(np.isfinite(best_points)):
        best_points = _build_hex_configuration(0.2, 0.42, (0.5, 0.5))
    
    return best_points
=======
def _nelder_mead_polish(points: np.ndarray, deadline: float) -> np.ndarray:
    """
    Nelder-Mead polish to squeeze out precision from a local optimum.
    """
    from scipy.optimize import minimize
    
    def objective(x):
        pts = x.reshape(13, 2)
        # Negate because we want to maximize
        return -_compute_score(pts)
    
    x0 = points.flatten()
    
    try:
        result = minimize(
            objective, x0, method='Nelder-Mead',
            options={'maxiter': 5000, 'xatol': 1e-10, 'fatol': 1e-12,
                     'adaptive': True}
        )
        if result.success or result.fun < objective(x0):
            polished = result.x.reshape(13, 2)
            if np.all(np.isfinite(polished)):
                return polished
    except Exception:
        pass
    
    return points


def heilbronn_convex13() -> np.ndarray:
    """
    Construct 13 points to maximize A_min / A_hull.
    Uses known hexagonal structure as the primary seed, then refines with
    fast local search and Nelder-Mead polish.
    """
    n = 13
    start_time = time.time()
    deadline = start_time + 580  # Safety margin
    
    best_points = None
    best_score = 0.0
    
    # Primary seed: known good hexagonal configuration
    configs = []
    
    # Hexagonal: center + 6 inner + 6 outer
    for r1 in [0.15, 0.18, 0.20, 0.22]:
        for r2 in [0.35, 0.40, 0.42, 0.45]:
            cfg = _build_hex_configuration(r1, r2, (0.5, 0.5))
            configs.append(cfg)
    
    # Concentric pentagon + heptagon
    configs.append(_build_concentric_config(13, (0.5, 0.5)))
    
    # Grid
    configs.append(_build_grid_config(13))
    
    # Random seeds
    for seed in range(10):
        configs.append(_build_random_config(13, seed))
    
    # Evaluate all configs
    scored = []
    for cfg in configs:
        if time.time() > deadline - 300:
            break
        s = _compute_score(cfg)
        scored.append((s, cfg))
    
    scored.sort(key=lambda x: -x[0])
    
    # Take top 5 for refinement
    top_configs = [cfg for _, cfg in scored[:5]]
    
    # Phase 1: Simulated annealing on top configs
    sa_deadline = deadline - 150
    for cfg in top_configs:
        if time.time() > sa_deadline:
            break
        sa_result = _simulated_annealing(cfg, sa_deadline,
                                          T_init=0.025, T_min=1e-9,
                                          cooling_rate=0.9992, max_iters=30000)
        s = _compute_score(sa_result)
        if s > best_score:
            best_score = s
            best_points = sa_result.copy()
    
    # Phase 2: Fast local search refinement
    if best_points is not None:
        ls_deadline = deadline - 80
        ls_result = _local_refine(best_points, ls_deadline, max_iters=100000)
        s = _compute_score(ls_result)
        if s > best_score:
            best_score = s
            best_points = ls_result.copy()
    
    # Phase 3: Nelder-Mead polish for high precision
    if best_points is not None and time.time() < deadline - 40:
        nm_result = _nelder_mead_polish(best_points, deadline - 20)
        s = _compute_score(nm_result)
        if s > best_score:
            best_score = s
            best_points = nm_result.copy()
    
    # Phase 4: Additional random restarts if time permits
    if best_points is not None and time.time() < deadline - 60:
        for _ in range(5):
            if time.time() > deadline - 40:
                break
            perturbed = best_points.copy()
            noise = np.random.randn(*perturbed.shape) * 0.03
            perturbed += noise
            np.clip(perturbed, 0, 1, out=perturbed)
            
            quick_ls = _local_refine(perturbed, deadline - 40, max_iters=20000)
            s = _compute_score(quick_ls)
            if s > best_score:
                best_score = s
                best_points = quick_ls.copy()
        
        # Final polish after restarts
        if time.time() < deadline - 20:
            nm_result = _nelder_mead_polish(best_points, deadline - 10)
            s = _compute_score(nm_result)
            if s > best_score:
                best_score = s
                best_points = nm_result.copy()
    
    # Safety
    if best_points is None or not np.all(np.isfinite(best_points)):
        best_points = _build_hex_configuration(0.2, 0.42, (0.5, 0.5))
    
    return best_points
>>>>>>> REPLACE
