<<<<<<< SEARCH
    # Optimization parameters
    min_step_size = 1e-8
    no_improve_count = 0
    max_no_improve = 40
    
    # Multi-stage optimization with clearer time allocation
    # Each stage gets a fraction of total time
    total_budget = deadline - start_time
    stages = [
        (0.03, 0.002, 0.40),   # (step_size, threshold, time_fraction)
        (0.01, 0.001, 0.30),
        (0.003, 0.0003, 0.20),
        (0.001, 0.0001, 0.10),
    ]
    
    for step_init, thresh_init, time_frac in stages:
        stage_end = start_time + total_budget * (1.0 - time_frac) + 5.0
        if time.time() >= stage_end:
            continue
        
        step_size = step_init
        threshold = thresh_init
        no_improve_count = 0
        
        while time.time() < stage_end:
            # Compute hull
            try:
                hull = ConvexHull(points)
            except Exception:
                # If hull fails, try a random kick
                kick = rng.normal(0, 0.02, points.shape)
                points = points + kick
                points /= np.linalg.norm(points, axis=1, keepdims=True)
                continue
            
            # Compute gradient
            grad = _gradient_of_max_circumradius(points, hull, threshold)
            
            # Normalize gradient
            grad_norm = np.linalg.norm(grad)
            if grad_norm < 1e-14:
                break
            
            # Update points
            new_points = points + step_size * grad
            new_points /= np.linalg.norm(new_points, axis=1, keepdims=True)
            
            current_radius = _compute_covering_radius(new_points)
            
            if current_radius < best_radius:
                best_radius = current_radius
                best_points = new_points.copy()
                points = new_points
                no_improve_count = 0
                # Slightly increase step size if making progress
                step_size = min(step_size * 1.01, step_init * 1.5)
            else:
                # If worse, revert and decrease step size
                step_size = max(step_size * 0.92, min_step_size)
                no_improve_count += 1
                
                if no_improve_count > max_no_improve:
                    # Stagnation: try a random kick
                    kick = rng.normal(0, 0.02, points.shape)
                    points = points + kick
                    points /= np.linalg.norm(points, axis=1, keepdims=True)
                    step_size = step_init * 0.5
                    no_improve_count = 0
                    current_radius = _compute_covering_radius(points)
                    if current_radius < best_radius:
                        best_radius = current_radius
                        best_points = points.copy()
=======
    # Optimization parameters
    min_step_size = 1e-8
    no_improve_count = 0
    max_no_improve = 60
    
    # Multi-stage optimization with clearer time allocation
    # Each stage gets a fraction of total time
    total_budget = deadline - start_time
    stages = [
        (0.03, 0.002, 0.35),   # (step_size, threshold, time_fraction)
        (0.01, 0.001, 0.25),
        (0.003, 0.0003, 0.20),
        (0.001, 0.0001, 0.10),
        (0.0003, 0.00005, 0.10),  # Added fine-tuning stage
    ]
    
    for step_init, thresh_init, time_frac in stages:
        # Calculate stage end time based on cumulative fractions
        # stages are processed in order, each consuming its fraction
        remaining_frac = time_frac
        for prev_step, prev_thresh, prev_frac in stages[:stages.index((step_init, thresh_init, time_frac))]:
            remaining_frac += prev_frac
        
        stage_end = start_time + total_budget * remaining_frac
        if time.time() >= stage_end:
            continue
        
        step_size = step_init
        threshold = thresh_init
        no_improve_count = 0
        consecutive_rejections = 0
        
        while time.time() < stage_end:
            # Compute hull
            try:
                hull = ConvexHull(points)
            except Exception:
                # If hull fails, try a random kick
                kick = rng.normal(0, 0.02, points.shape)
                points = points + kick
                points /= np.linalg.norm(points, axis=1, keepdims=True)
                continue
            
            # Compute gradient
            grad = _gradient_of_max_circumradius(points, hull, threshold)
            
            # Normalize gradient
            grad_norm = np.linalg.norm(grad)
            if grad_norm < 1e-14:
                break
            
            # Adaptive step size: use line search-like approach
            # Try the step, if it fails, halve the step
            accepted = False
            trial_step = step_size
            for _ in range(5):
                new_points = points + trial_step * grad
                new_points /= np.linalg.norm(new_points, axis=1, keepdims=True)
                
                current_radius = _compute_covering_radius(new_points)
                
                if current_radius < best_radius:
                    best_radius = current_radius
                    best_points = new_points.copy()
                    points = new_points
                    no_improve_count = 0
                    consecutive_rejections = 0
                    accepted = True
                    # Slightly increase step size if making progress
                    step_size = min(step_size * 1.02, step_init * 2.0)
                    break
                else:
                    trial_step *= 0.5
                    if trial_step < min_step_size:
                        break
            
            if not accepted:
                no_improve_count += 1
                consecutive_rejections += 1
                step_size = max(step_size * 0.90, min_step_size)
                
                if no_improve_count > max_no_improve or consecutive_rejections > 10:
                    # Stagnation: try a random kick
                    kick = rng.normal(0, 0.015, points.shape)
                    points = points + kick
                    points /= np.linalg.norm(points, axis=1, keepdims=True)
                    step_size = step_init * 0.5
                    no_improve_count = 0
                    consecutive_rejections = 0
                    current_radius = _compute_covering_radius(points)
                    if current_radius < best_radius:
                        best_radius = current_radius
                        best_points = points.copy()
>>>>>>> REPLACE

