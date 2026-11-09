<<<<<<< SEARCH
    # Phase 1: Lloyd's algorithm to get a good local optimum
    lloyd_alpha = 0.5
    for step in range(200):
        if time.time() - start_time > time_limit:
            break
        points = _lloyd_step(points, lloyd_alpha)
        if step % 20 == 0:
            cr = _covering_radius(points)
            if cr < best_cr:
                best_cr = cr
                best_points = points.copy()
        # Decay alpha
        lloyd_alpha *= 0.99
    
    # Phase 2: Simulated Annealing for global exploration
    # Start temperature based on current covering radius
    temp = best_cr * 0.5
    min_temp = 1e-4
    cooling_rate = 0.995
    
    sa_points = best_points.copy()
    sa_cr = best_cr
    
    while time.time() - start_time < time_limit:
        new_points = _simulated_annealing_step(sa_points, temp, rng)
        
        # Evaluate new configuration
        new_cr = _covering_radius(new_points)
        
        # Accept if better, or with probability exp(-(new_cr - sa_cr)/temp)
        if new_cr < sa_cr:
            sa_points = new_points
            sa_cr = new_cr
            if sa_cr < best_cr:
                best_cr = sa_cr
                best_points = sa_points.copy()
        else:
            delta = new_cr - sa_cr
            if delta < 1e-10:
                sa_points = new_points
                sa_cr = new_cr
            else:
                accept_prob = np.exp(-delta / max(temp, 1e-10))
                if rng.random() < accept_prob:
                    sa_points = new_points
                    sa_cr = new_cr
        
        # Cool down
        temp = max(temp * cooling_rate, min_temp)
        
        # Periodically re-run Lloyd from current SA state
        if int((time.time() - start_time) * 10) % 50 == 0 and time.time() - start_time < time_limit - 10:
            lloyd_points = sa_points.copy()
            for lstep in range(30):
                if time.time() - start_time > time_limit:
                    break
                lloyd_points = _lloyd_step(lloyd_points, 0.3)
            lloyd_cr = _covering_radius(lloyd_points)
            if lloyd_cr < best_cr:
                best_cr = lloyd_cr
                best_points = lloyd_points.copy()
            # Update SA state with improved Lloyd result
            if lloyd_cr < sa_cr:
                sa_points = lloyd_points.copy()
                sa_cr = lloyd_cr
    
    # Final Lloyd refinement
    if time.time() - start_time < time_limit - 5:
        final_points = best_points.copy()
        for fstep in range(50):
            if time.time() - start_time > time_limit:
                break
            final_points = _lloyd_step(final_points, 0.2)
        final_cr = _covering_radius(final_points)
        if final_cr < best_cr:
            best_cr = final_cr
            best_points = final_points.copy()
    
    return best_points
=======
    # Phase 1: Lloyd's algorithm to get a good local optimum
    lloyd_alpha = 0.5
    for step in range(150):
        if time.time() - start_time > time_limit:
            break
        points = _lloyd_step(points, lloyd_alpha)
        if step % 10 == 0:
            cr = _covering_radius(points)
            if cr < best_cr:
                best_cr = cr
                best_points = points.copy()
        # Decay alpha
        lloyd_alpha *= 0.98
    
    # Phase 2: Riemannian gradient flow to minimize worst facet circumradius
    rg_points = best_points.copy()
    rg_cr = best_cr
    step_size = 0.05
    
    while time.time() - start_time < time_limit - 5:
        new_points = _riemannian_gradient_step(rg_points, step_size)
        new_cr, _, _ = _covering_radius_and_worst(new_points)
        
        if new_cr < rg_cr - 1e-12:
            rg_points = new_points
            rg_cr = new_cr
            if rg_cr < best_cr:
                best_cr = rg_cr
                best_points = rg_points.copy()
            # Decrease step size slightly as we converge
            step_size *= 0.995
        else:
            # If no improvement, reduce step size and try again
            step_size *= 0.9
            if step_size < 1e-4:
                # Reset with a small perturbation to escape local minimum
                perturbed = rg_points.copy()
                for idx in range(N):
                    perturbation = rng.standard_normal(3) * 0.01
                    perturbed[idx] = rg_points[idx] + perturbation
                    norm = np.linalg.norm(perturbed[idx])
                    if norm > 1e-12:
                        perturbed[idx] = perturbed[idx] / norm
                perturbed_cr, _, _ = _covering_radius_and_worst(perturbed)
                if perturbed_cr < rg_cr:
                    rg_points = perturbed
                    rg_cr = perturbed_cr
                step_size = 0.05
    
    # Phase 3: Final Lloyd refinement
    if time.time() - start_time < time_limit - 3:
        final_points = best_points.copy()
        for fstep in range(30):
            if time.time() - start_time > time_limit:
                break
            final_points = _lloyd_step(final_points, 0.1)
        final_cr = _covering_radius(final_points)
        if final_cr < best_cr:
            best_cr = final_cr
            best_points = final_points.copy()
    
    # Phase 4: Final Riemannian polish
    if time.time() - start_time < time_limit - 1:
        polish_points = best_points.copy()
        polish_cr = best_cr
        for _ in range(50):
            if time.time() - start_time > time_limit:
                break
            new_points = _riemannian_gradient_step(polish_points, 0.01)
            new_cr, _, _ = _covering_radius_and_worst(new_points)
            if new_cr < polish_cr - 1e-14:
                polish_points = new_points
                polish_cr = new_cr
            else:
                break
        if polish_cr < best_cr:
            best_cr = polish_cr
            best_points = polish_points.copy()
    
    return best_points
>>>>>>> REPLACE
