def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    
    best_area = 0.0
    best_bary = None
    
    # Define bounds for barycentric coordinates [0, 1]
    bounds = [(0.0, 1.0)] * (N * 3)
    
    # Global Search with Differential Evolution
    try:
        de_result = differential_evolution(
            _objective_de,
            bounds,
            maxiter=100,
            popsize=5,
            tol=1e-6,
            seed=42,
            polish=False,
            update_interval=1
        )
        
        if time.time() > deadline - 10.0:
             candidate_bary = _project_bary(de_result.x.reshape(N, 3))
             area = _min_area_bary(candidate_bary)
             if area > best_area:
                 best_area = area
                 best_bary = candidate_bary
    except Exception:
        pass
    
    # Local Refinement with Nelder-Mead
    candidates = []
    
    if 'de_result' in locals() and de_result.success:
        candidates.append(de_result.x)
    
    # Add some random candidates
    rng = np.random.default_rng(123)
    for _ in range(5):
        candidates.append(rng.uniform(0, 1, size=(N, 3)).flatten())
        
    # Also add a symmetric-ish start
    center = np.array([1/3, 1/3, 1/3])
    sym_init = np.tile(center, (N, 1))
    sym_init += rng.normal(0, 0.1, size=(N, 3))
    candidates.append(sym_init.flatten())
    
    for x0 in candidates:
        if time.time() > deadline - 5.0:
            break
            
        try:
            res = minimize(
                _objective_local,
                x0,
                method='Nelder-Mead',
                options={'maxiter': 1000, 'xatol': 1e-8, 'fatol': 1e-10}
            )
            
            refined_bary = _project_bary(res.x.reshape(N, 3))
            true_area = _min_area_bary(refined_bary)
            
            if true_area > best_area:
                best_area = true_area
                best_bary = refined_bary.copy()
                
        except Exception:
            continue
            
    if best_bary is None:
        rng = np.random.default_rng(0)
        best_bary = rng.dirichlet([1, 1, 1], size=N)
        best_bary = _project_bary(best_bary)
        
    return _bary_to_cart(best_bary)
