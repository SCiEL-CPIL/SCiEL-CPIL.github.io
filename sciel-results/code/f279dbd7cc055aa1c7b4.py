<<<<<<< SEARCH
def construct():
    start_time = time.time()
    DEADLINE = 145.0
    safety_margin = 5.0
    
    n = 11
    outer_angle = 30.0
    
    # Seed solution
    seed_inner, seed_center, seed_side, seed_angle = _seed_solution()
    best_inner = np.array(seed_inner)
    best_side = seed_side
    
    # Initial Nelder-Mead to get a good starting point
    init_params = best_inner.flatten()
    try:
        res = minimize(_objective, init_params, args=(outer_angle, n), 
                      method='Nelder-Mead', 
                      options={'maxiter': 300, 'xatol': 1e-6, 'fatol': 1e-6})
        if res.fun < best_side:
            candidate = res.x.reshape(n, 3)
            candidate_side = _compute_side(candidate, outer_angle)
            if _is_valid(candidate, candidate_side, outer_angle, min_sep=1e-9):
                best_inner = candidate
                best_side = candidate_side
    except Exception:
        pass
    
    # PSO phase
    remaining = DEADLINE - safety_margin - (time.time() - start_time)
    if remaining > 10:
        pso_time = min(remaining * 0.5, 40.0)
        try:
            pso_result, pso_val = _psp_swarm(best_inner, outer_angle, n, iterations=200, swarm_size=25, time_limit=pso_time)
            if pso_val < best_side:
                candidate = pso_result
                candidate_side = _compute_side(candidate, outer_angle)
                if _is_valid(candidate, candidate_side, outer_angle, min_sep=1e-9):
                    best_inner = candidate
                    best_side = candidate_side
        except Exception:
            pass
    
    # Local refinement phase
    remaining = DEADLINE - safety_margin - (time.time() - start_time)
    if remaining > 5:
        try:
            refined = _local_refine(best_inner, outer_angle, n, time_limit=min(remaining * 0.8, 30.0))
            refined_side = _compute_side(refined, outer_angle)
            if refined_side < best_side and _is_valid(refined, refined_side, outer_angle, min_sep=1e-9):
                best_inner = refined
                best_side = refined_side
        except Exception:
            pass
    
    # Final check
    final_side = _compute_side(best_inner, outer_angle)
    if not _is_valid(best_inner, final_side, outer_angle, min_sep=1e-9):
        # Fallback to seed
        best_inner = np.array(seed_inner)
        final_side = seed_side
    
    # Format output
    inner_rows = []
    for row in best_inner:
        inner_rows.append([float(row[0]), float(row[1]), float(row[2])])
        
    center = (0.0, 0.0)
    side = float(final_side)
    angle_deg = float(outer_angle)
    
    return inner_rows, center, side, angle_deg
=======
def construct():
    start_time = time.time()
    DEADLINE = 145.0
    safety_margin = 5.0
    
    n = 11
    
    # Seed solution
    seed_inner, seed_center, seed_side, seed_angle = _seed_solution()
    best_inner = np.array(seed_inner)
    best_side = seed_side
    best_outer_angle = seed_angle
    
    # Optimize outer angle for the seed configuration
    elapsed = time.time() - start_time
    if DEADLINE - safety_margin - elapsed > 3.0:
        best_outer_angle, best_side = _optimize_outer_angle(best_inner, n, time_limit=min(3.0, DEADLINE - safety_margin - elapsed))
    
    # Phase 1: Nelder-Mead on inner params with fixed outer angle
    remaining = DEADLINE - safety_margin - (time.time() - start_time)
    if remaining > 8:
        try:
            res = minimize(_objective, best_inner.flatten(), args=(best_outer_angle, n), 
                          method='Nelder-Mead', 
                          options={'maxiter': 500, 'xatol': 1e-7, 'fatol': 1e-7, 'adaptive': True})
            candidate = res.x.reshape(n, 3)
            candidate_side = _compute_side(candidate, best_outer_angle)
            if candidate_side < best_side and _is_valid(candidate, candidate_side, best_outer_angle, min_sep=1e-9):
                best_inner = candidate
                best_side = candidate_side
        except Exception:
            pass
    
    # Re-optimize outer angle after inner improvement
    remaining = DEADLINE - safety_margin - (time.time() - start_time)
    if remaining > 3.0:
        new_angle, new_side = _optimize_outer_angle(best_inner, n, time_limit=min(3.0, remaining))
        if new_side < best_side:
            best_outer_angle = new_angle
            best_side = new_side
    
    # Phase 2: PSO
    remaining = DEADLINE - safety_margin - (time.time() - start_time)
    if remaining > 15:
        pso_time = min(remaining * 0.4, 35.0)
        try:
            pso_result, pso_val = _psp_swarm(best_inner, best_outer_angle, n, iterations=250, swarm_size=20, time_limit=pso_time)
            candidate_side = _compute_side(pso_result, best_outer_angle)
            if candidate_side < best_side and _is_valid(pso_result, candidate_side, best_outer_angle, min_sep=1e-9):
                best_inner = pso_result
                best_side = candidate_side
        except Exception:
            pass
    
    # Re-optimize outer angle after PSO
    remaining = DEADLINE - safety_margin - (time.time() - start_time)
    if remaining > 3.0:
        new_angle, new_side = _optimize_outer_angle(best_inner, n, time_limit=min(3.0, remaining))
        if new_side < best_side:
            best_outer_angle = new_angle
            best_side = new_side
    
    # Phase 3: Final local refinement
    remaining = DEADLINE - safety_margin - (time.time() - start_time)
    if remaining > 5:
        try:
            refined = _local_refine(best_inner, best_outer_angle, n, time_limit=min(remaining * 0.8, 25.0))
            refined_side = _compute_side(refined, best_outer_angle)
            if refined_side < best_side and _is_valid(refined, refined_side, best_outer_angle, min_sep=1e-9):
                best_inner = refined
                best_side = refined_side
        except Exception:
            pass
    
    # Final outer angle optimization
    remaining = DEADLINE - safety_margin - (time.time() - start_time)
    if remaining > 2.0:
        new_angle, new_side = _optimize_outer_angle(best_inner, n, time_limit=min(2.0, remaining))
        if new_side < best_side:
            best_outer_angle = new_angle
            best_side = new_side
    
    # Final check
    final_side = _compute_side(best_inner, best_outer_angle)
    if not _is_valid(best_inner, final_side, best_outer_angle, min_sep=1e-9):
        # Fallback to seed
        best_inner = np.array(seed_inner)
        final_side = seed_side
        best_outer_angle = seed_angle
    
    # Format output
    inner_rows = []
    for row in best_inner:
        inner_rows.append([float(row[0]), float(row[1]), float(row[2])])
        
    center = (0.0, 0.0)
    side = float(final_side)
    angle_deg = float(best_outer_angle)
    
    return inner_rows, center, side, angle_deg
>>>>>>> REPLACE
