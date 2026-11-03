<<<<<<< SEARCH
def _multi_start_local_search(n_starts=8, time_budget=60.0):
    """
    Multi-start local search: try multiple initial configurations and polish each.
    Returns the best centers and score found.
    """
    start_time = time.time()
    best_centers = None
    best_score = 0.0
    
    # Generate diverse initial configurations
    inits = []
    
    # Base ring configuration
    base_radii = np.array([0.252, 0.454, 0.655, 0.855])
    base_angles = np.array([0.0, np.pi / 12, np.pi / 6, 0.0])
    base_params = np.concatenate([base_radii, base_angles])
    inits.append(_build_points_from_params(base_params))
    
    # Perturbed ring configurations with varying perturbation scales
    rng = np.random.RandomState(42)
    perturbation_scales = [0.03, 0.05, 0.08, 0.12, 0.15, 0.20, 0.30]
    for k in range(n_starts - 1):
        params = base_params.copy()
        scale = perturbation_scales[k % len(perturbation_scales)]
        params[:4] += rng.randn(4) * scale
        params[:4] = np.clip(np.abs(params[:4]), 0.1, 0.95)
        params[4:] += rng.randn(4) * scale * 6.0
        inits.append(_build_points_from_params(params))
    
    for init in inits:
        if time.time() - start_time > time_budget:
            break
        
        # Smooth polish with two beta values
        polished = _polish_with_smooth(init, beta=80.0, maxiter=150)
        score = _compute_score(polished)
        
        if score > best_score:
            best_score = score
            best_centers = polished.copy()
        
        # Second polish with higher beta if time allows
        if time.time() - start_time < time_budget - 5.0:
            polished2 = _polish_with_smooth(polished, beta=200.0, maxiter=100)
            score2 = _compute_score(polished2)
            if score2 > best_score:
                best_score = score2
                best_centers = polished2.copy()
    
    return best_centers, best_score
=======
def _multi_start_local_search(n_starts=8, time_budget=60.0):
    """
    Multi-start local search: try multiple initial configurations and polish each.
    Returns the best centers and score found.
    """
    start_time = time.time()
    best_centers = None
    best_score = 0.0
    
    # Generate diverse initial configurations
    inits = []
    
    # Base ring configuration
    base_radii = np.array([0.252, 0.454, 0.655, 0.855])
    base_angles = np.array([0.0, np.pi / 12, np.pi / 6, 0.0])
    base_params = np.concatenate([base_radii, base_angles])
    inits.append(_build_points_from_params(base_params))
    
    # Perturbed ring configurations with varying perturbation scales
    rng = np.random.RandomState(42)
    perturbation_scales = [0.03, 0.05, 0.08, 0.12, 0.15, 0.20, 0.30]
    for k in range(n_starts - 1):
        params = base_params.copy()
        scale = perturbation_scales[k % len(perturbation_scales)]
        params[:4] += rng.randn(4) * scale
        params[:4] = np.clip(np.abs(params[:4]), 0.1, 0.95)
        params[4:] += rng.randn(4) * scale * 6.0
        inits.append(_build_points_from_params(params))
    
    for init in inits:
        if time.time() - start_time > time_budget:
            break
        
        # Multi-stage smooth polish with increasing beta
        polished = _polish_with_smooth(init, beta=50.0, maxiter=200)
        score = _compute_score(polished)
        
        if score > best_score:
            best_score = score
            best_centers = polished.copy()
        
        if time.time() - start_time < time_budget - 5.0:
            polished2 = _polish_with_smooth(polished, beta=200.0, maxiter=200)
            score2 = _compute_score(polished2)
            if score2 > best_score:
                best_score = score2
                best_centers = polished2.copy()
        
        # Constraint push for local refinement
        if time.time() - start_time < time_budget - 3.0:
            pushed, pushed_score = _constraint_push(polished2, n_iters=100)
            if pushed_score > best_score:
                best_score = pushed_score
                best_centers = pushed.copy()
    
    return best_centers, best_score
>>>>>>> REPLACE

