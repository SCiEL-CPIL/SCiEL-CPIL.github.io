<<<<<<< SEARCH
def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # 5s safety margin
    
    # 1. Initialization
    best_pos = _generate_initial_positions()
    
    # Initial soft-min optimization to spread out spheres evenly
    # This helps escape the initial lattice artifacts
    try:
        soft_pos, _ = _minimize_soft_min(best_pos, beta=50.0, max_iter=200)
        soft_r = _compute_radius(soft_pos)
        if soft_r > _compute_radius(best_pos):
            best_pos = soft_pos
    except Exception:
        pass
    
    # Initial optimization with a conservative radius using energy minimization
    initial_r = _compute_radius(best_pos)
    if initial_r < 0.08:
        initial_r = 0.08
    else:
        initial_r = initial_r * 0.95  # Start slightly below current to ensure convergence
        
    best_pos, energy, _ = _minimize_energy(best_pos, initial_r, max_iter=300)
    best_r = _compute_radius(best_pos)
    
    if best_r <= 0:
        rng = np.random.default_rng(123)
        best_pos = rng.uniform(-0.45, 0.45, size=(N, 3))
        best_r = _compute_radius(best_pos)
    
    # 2. Iterative Radius Increase with adaptive step size
    current_r = best_r
    step_size = 0.001  # Start with a reasonable step
    
    consecutive_failures = 0
    max_failures = 3
    
    rng_search = np.random.default_rng(123)
    
    while time.time() < deadline - 10.0:
        target_r = current_r + step_size
        
        if consecutive_failures >= max_failures:
            step_size *= 0.6
            consecutive_failures = 0
            if step_size < 1e-8:
                break
        
        # Try multiple perturbations to escape local minima
        candidates = [best_pos.copy()]
        for scale in [0.003, 0.01, 0.03]:
            perturbed = best_pos + rng_search.normal(0, scale, size=(N, 3))
            perturbed = np.clip(perturbed, -0.5, 0.5)
            candidates.append(perturbed)
        
        success = False
        for start_pos in candidates:
            # Increase max_iter for higher precision as we get closer to optimum
            max_iter = 400 if current_r < 0.105 else 600
            
            new_pos, energy, opt_success = _minimize_energy(start_pos, target_r, max_iter=max_iter)
            
            if energy < 1e-10:
                actual_r = _compute_radius(new_pos)
                if actual_r >= target_r - 1e-7:
                    best_pos = new_pos
                    best_r = actual_r
                    current_r = actual_r
                    success = True
                    consecutive_failures = 0
                    break
            else:
                # Even if energy is not perfectly zero, if we improved the radius, accept it
                actual_r = _compute_radius(new_pos)
                if actual_r > current_r + 1e-8:
                    best_pos = new_pos
                    best_r = actual_r
                    current_r = actual_r
                    # Don't mark as full success, but accept the improvement
                    pass
        
        if not success:
            consecutive_failures += 1
            
            # If we failed, try a soft-min step to re-spread and escape local minimum
            if time.time() < deadline - 15.0 and consecutive_failures % 2 == 0:
                try:
                    # Use a lower beta for more exploration
                    soft_pos, _ = _minimize_soft_min(best_pos, beta=30.0, max_iter=100)
                    soft_r = _compute_radius(soft_pos)
                    if soft_r > best_r:
                        best_pos = soft_pos
                        best_r = soft_r
                        current_r = best_r
                    else:
                        # If soft-min didn't help, try polishing the current best position
                        polished, _, _ = _minimize_energy(best_pos, current_r, max_iter=300)
                        polished_r = _compute_radius(polished)
                        if polished_r > best_r:
                            best_pos = polished
                            best_r = polished_r
                            current_r = best_r
                except Exception:
                    pass

    # 3. Final High-Precision Polish
    if time.time() < deadline - 3.0:
        # Try to push slightly beyond current_r
        final_target = best_r + 5e-6
        polished_pos, final_energy, _ = _minimize_energy(best_pos, final_target, max_iter=800, ftol=1e-15, gtol=1e-14)
        final_r = _compute_radius(polished_pos)
        
        if final_r > best_r:
            best_pos = polished_pos
            best_r = final_r
            
        # One more polish at the exact best_r to ensure tightness
        if time.time() < deadline - 1.0:
            polished_pos, _, _ = _minimize_energy(best_pos, best_r, max_iter=500, ftol=1e-15, gtol=1e-14)
            final_r = _compute_radius(polished_pos)
            if final_r > best_r:
                best_pos = polished_pos
                best_r = final_r

    return best_pos
=======
def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # 5s safety margin
    
    # 1. Initialization
    best_pos = _generate_initial_positions()
    best_r = _compute_radius(best_pos)
    
    # Initial polish to get a good starting point
    try:
        polished, _, _ = _minimize_energy(best_pos, best_r, max_iter=200)
        polished_r = _compute_radius(polished)
        if polished_r > best_r:
            best_pos = polished
            best_r = polished_r
    except Exception:
        pass

    # 2. Simulated Annealing Phase
    # SA is better at global exploration than pure gradient descent
    current_pos = best_pos.copy()
    current_r = best_r
    
    # Temperature schedule
    T_start = 0.05
    T_end = 1e-6
    total_steps = 50000  # Adjust based on time budget
    steps_per_sec = 10000 # Estimate
    max_steps = int((deadline - time.time() - 5.0) * steps_per_sec)
    max_steps = min(total_steps, max_steps)
    
    if max_steps <= 0:
        return best_pos

    alpha = (T_end / T_start) ** (1.0 / max_steps)
    T = T_start
    
    rng = np.random.default_rng(42)
    
    last_polish_time = time.time()
    polish_interval = 5.0 # seconds
    
    for i in range(max_steps):
        if time.time() > deadline - 5.0:
            break
            
        # Generate a move
        # Randomly select a subset of spheres to move to allow larger moves
        # or move all spheres with small noise
        if rng.random() < 0.1:
            # Large jump for a random subset
            idx = rng.choice(N, size=N//5, replace=False)
            move_scale = 0.05
        else:
            # Small jump for all spheres
            idx = np.arange(N)
            move_scale = 0.005 + 0.01 * (T / T_start)
            
        delta = rng.normal(0, move_scale, size=(N, 3))
        new_pos = current_pos.copy()
        new_pos[idx] += delta
        new_pos = np.clip(new_pos, -0.5, 0.5)
        
        new_r = _compute_radius(new_pos)
        
        # Acceptance criterion
        if new_r > current_r:
            current_pos = new_pos
            current_r = new_r
            if current_r > best_r:
                best_pos = current_pos.copy()
                best_r = current_r
        else:
            delta_r = current_r - new_r
            # Accept if exp(-delta_r / T) > random
            if rng.random() < np.exp(-delta_r / T):
                current_pos = new_pos
                current_r = new_r
        
        T *= alpha
        
        # Periodic Polish
        if time.time() - last_polish_time > polish_interval:
            if time.time() < deadline - 10.0:
                try:
                    # Polish the best found so far
                    polished, _, _ = _minimize_energy(best_pos, best_r, max_iter=100)
                    polished_r = _compute_radius(polished)
                    if polished_r > best_r:
                        best_pos = polished
                        best_r = polished_r
                        # Update current state to the polished best to continue SA from a better point
                        current_pos = best_pos.copy()
                        current_r = best_r
                except Exception:
                    pass
            last_polish_time = time.time()

    # 3. Final High-Precision Polish
    if time.time() < deadline - 2.0:
        try:
            # Use the energy minimizer to push slightly beyond best_r
            # This acts as a local refinement
            target_r = best_r + 1e-5
            polished_pos, energy, _ = _minimize_energy(best_pos, target_r, max_iter=500, ftol=1e-14, gtol=1e-12)
            final_r = _compute_radius(polished_pos)
            if final_r > best_r:
                best_pos = polished_pos
                best_r = final_r
                
            # Final tight polish
            if time.time() < deadline - 0.5:
                polished_pos, _, _ = _minimize_energy(best_pos, best_r, max_iter=300, ftol=1e-15, gtol=1e-14)
                final_r = _compute_radius(polished_pos)
                if final_r > best_r:
                    best_pos = polished_pos
                    best_r = final_r
        except Exception:
            pass

    return best_pos
>>>>>>> REPLACE
