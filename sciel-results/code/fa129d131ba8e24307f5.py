<<<<<<< SEARCH
def construct():
    start_time = time.time()
    deadline = start_time + 120  # 120s limit, keep safety margin
    
    # Generate multiple initializations
    inits = []
    
    # Grid initialization
    inits.append(_make_initial_grid())
    
    # Greedy best-fit
    inits.append(_greedy_best_fit())
    
    # Random perturbations of grid
    rng = np.random.RandomState(42)
    for _ in range(3):
        grid = _make_initial_grid()
        # Perturb positions slightly
        perturbed = grid.copy()
        perturbed[:, 0] += rng.uniform(-0.02, 0.02, N)
        perturbed[:, 1] += rng.uniform(-0.02, 0.02, N)
        # Clip to [0,1]
        perturbed[:, 0] = np.clip(perturbed[:, 0], 0, 1)
        perturbed[:, 1] = np.clip(perturbed[:, 1], 0, 1)
        inits.append(perturbed)
    
    best_result = None
    best_score = -1
    
    for init in inits:
        if time.time() > deadline - 30:  # Keep 30s for polish
            break
        
        # Try SLSQP optimization
        try:
            result = _optimize_slsqp(init, max_iter=300, tol=1e-8)
            if _valid(result):
                score = _score(result)
                if score > best_score:
                    best_score = score
                    best_result = result
        except Exception:
            pass
        
        # Try penalty method as backup
        if time.time() > deadline - 30:
            break
        try:
            result = _penalty_optimize(init, max_iter=300, mu=50.0, tol=1e-8)
            if _valid(result):
                score = _score(result)
                if score > best_score:
                    best_score = score
                    best_result = result
        except Exception:
            pass
    
    # If no valid result yet, use the grid
    if best_result is None:
        best_result = _make_initial_grid()
        best_score = _score(best_result)
    
    # Final polish
    if time.time() < deadline - 5:
        try:
            polished = _final_polish(best_result, time_budget=min(25, deadline - time.time() - 5))
            if _valid(polished):
                polished_score = _score(polished)
                if polished_score > best_score:
                    best_result = polished
                    best_score = polished_score
        except Exception:
            pass
    
    return best_result
=======
def construct():
    start_time = time.time()
    deadline = start_time + 115  # 115s limit, keep 5s safety margin
    
    # Generate multiple initializations
    inits = []
    
    # Grid initialization
    inits.append(_make_initial_grid())
    
    # Hexagonal initialization
    inits.append(_make_hexagonal())
    
    # Greedy best-fit
    inits.append(_greedy_best_fit())
    
    # Random perturbations of grid
    rng = np.random.RandomState(42)
    for _ in range(2):
        grid = _make_initial_grid()
        # Perturb positions slightly
        perturbed = grid.copy()
        perturbed[:, 0] += rng.uniform(-0.02, 0.02, N)
        perturbed[:, 1] += rng.uniform(-0.02, 0.02, N)
        # Clip to [0,1]
        perturbed[:, 0] = np.clip(perturbed[:, 0], 0, 1)
        perturbed[:, 1] = np.clip(perturbed[:, 1], 0, 1)
        inits.append(perturbed)
    
    # Random perturbations of hexagonal
    for _ in range(2):
        hex_init = _make_hexagonal()
        perturbed = hex_init.copy()
        perturbed[:, 0] += rng.uniform(-0.015, 0.015, N)
        perturbed[:, 1] += rng.uniform(-0.015, 0.015, N)
        # Clip to [0,1]
        perturbed[:, 0] = np.clip(perturbed[:, 0], 0, 1)
        perturbed[:, 1] = np.clip(perturbed[:, 1], 0, 1)
        inits.append(perturbed)
    
    best_result = None
    best_score = -1
    
    for init in inits:
        if time.time() > deadline - 35:  # Keep 35s for polish
            break
        
        # Try SLSQP optimization
        try:
            result = _optimize_slsqp(init, max_iter=200, tol=1e-8)
            if _valid(result):
                score = _score(result)
                if score > best_score:
                    best_score = score
                    best_result = result
        except Exception:
            pass
        
        # Try penalty method as backup
        if time.time() > deadline - 35:
            break
        try:
            result = _penalty_optimize(init, max_iter=200, mu=50.0, tol=1e-8)
            if _valid(result):
                score = _score(result)
                if score > best_score:
                    best_score = score
                    best_result = result
        except Exception:
            pass
    
    # If no valid result yet, use the grid
    if best_result is None:
        best_result = _make_initial_grid()
        best_score = _score(best_result)
    
    # Final polish
    if time.time() < deadline - 3:
        try:
            remaining = deadline - time.time() - 3
            polished = _final_polish(best_result, time_budget=min(30, remaining))
            if _valid(polished):
                polished_score = _score(polished)
                if polished_score > best_score:
                    best_result = polished
                    best_score = polished_score
        except Exception:
            pass
    
    return best_result
>>>>>>> REPLACE
