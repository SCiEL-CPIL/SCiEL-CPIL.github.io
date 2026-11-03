<<<<<<< SEARCH
def construct():
    start_time = time.time()
    
    # Generate diverse initial points
    starts = []
    starts.append(_generate_grid_start())
    starts.append(_generate_fcc_start())
    starts.append(_generate_bcc_start())
    starts.append(_generate_hcp_start())
    for i in range(3):
        starts.append(_generate_random_start(100 + i))
    
    best_pts = None
    best_score = -np.inf
    
    # Run SA on each start, allocating time
    num_runs = len(starts)
    time_per_run = (DEADLINE - (time.time() - start_time) - 25) / num_runs  # Reserve 25s for polish
    
    for i, init_pts in enumerate(starts):
        if time.time() - start_time > DEADLINE - 25:
            break
        
        # Run SA
        sa_pts = _sa_optimize(init_pts, time_per_run, seed=1000 + i)
        sa_score = _compute_radius(sa_pts)
        
        if sa_score > best_score:
            best_score = sa_score
            best_pts = sa_pts
    
    # Polish the best result with L-BFGS-B
    if best_pts is not None and time.time() - start_time < DEADLINE - 12:
        remaining_time = DEADLINE - 12 - (time.time() - start_time)
        polished_pts = _polish_lbfgs(best_pts, remaining_time)
        polished_score = _compute_radius(polished_pts)
        
        if polished_score > best_score:
            best_pts = polished_pts
            best_score = polished_score
    
    # Active-set refinement for final precision
    if best_pts is not None and time.time() - start_time < DEADLINE - 5:
        remaining_time = DEADLINE - 5 - (time.time() - start_time)
        refined_pts = _active_set_refine(best_pts, remaining_time)
        refined_score = _compute_radius(refined_pts)
        
        if refined_score > best_score:
            best_pts = refined_pts
            best_score = refined_score
    
    # Fallback
    if best_pts is None:
        best_pts = starts[0]
    
    # Ensure validity
    best_pts = np.clip(best_pts, -0.49999999, 0.49999999)
    
    return best_pts
=======
def construct():
    start_time = time.time()
    
    # Generate diverse initial points
    starts = []
    starts.append(_generate_grid_start())
    starts.append(_generate_fcc_start())
    starts.append(_generate_bcc_start())
    starts.append(_generate_hcp_start())
    for i in range(3):
        starts.append(_generate_random_start(100 + i))
    
    best_pts = None
    best_score = -np.inf
    
    # Run SA on each start, allocating time
    num_runs = len(starts)
    # Reserve 35s for polish + active set
    time_for_sa = DEADLINE - (time.time() - start_time) - 35
    time_per_run = time_for_sa / num_runs
    
    sa_results = []
    for i, init_pts in enumerate(starts):
        if time.time() - start_time > DEADLINE - 35:
            break
        
        # Run SA
        sa_pts = _sa_optimize(init_pts, time_per_run, seed=1000 + i)
        sa_score = _compute_radius(sa_pts)
        sa_results.append((sa_score, sa_pts))
        
        if sa_score > best_score:
            best_score = sa_score
            best_pts = sa_pts
    
    # Sort SA results by score, polish top 3
    sa_results.sort(key=lambda x: x[0], reverse=True)
    
    polished_best = best_pts
    polished_best_score = best_score
    
    for rank, (sa_score, sa_pts) in enumerate(sa_results[:3]):
        if time.time() - start_time > DEADLINE - 10:
            break
        remaining_time = min(15, DEADLINE - 10 - (time.time() - start_time))
        if remaining_time < 3:
            break
        try:
            polished_pts = _polish_lbfgs(sa_pts, remaining_time)
            polished_score = _compute_radius(polished_pts)
            if polished_score > polished_best_score:
                polished_best_score = polished_score
                polished_best = polished_pts
        except Exception:
            pass
    
    if polished_best_score > best_score:
        best_pts = polished_best
        best_score = polished_best_score
    
    # Active-set refinement for final precision
    if best_pts is not None and time.time() - start_time < DEADLINE - 4:
        remaining_time = DEADLINE - 4 - (time.time() - start_time)
        refined_pts = _active_set_refine(best_pts, remaining_time)
        refined_score = _compute_radius(refined_pts)
        
        if refined_score > best_score:
            best_pts = refined_pts
            best_score = refined_score
    
    # Fallback
    if best_pts is None:
        best_pts = starts[0]
    
    # Ensure validity
    best_pts = np.clip(best_pts, -0.49999999, 0.49999999)
    
    return best_pts
>>>>>>> REPLACE

