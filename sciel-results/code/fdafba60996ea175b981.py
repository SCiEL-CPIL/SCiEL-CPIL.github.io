<<<<<<< SEARCH
def _simulated_annealing(points_init, time_limit, rng):
    """
    Run Simulated Annealing to maximize the min triangle area.
    """
    start_time = time.time()
    current_points = points_init.copy()
    current_area = _min_triangle_area_fast(current_points)
    
    best_points = current_points.copy()
    best_area = current_area
    
    # Initial temperature
    T = 0.01
    # Cooling rate
    alpha = 0.999
    # Step size
    step_size = 0.05
    
    iterations = 0
    while time.time() < time_limit:
        # Generate a new candidate
        # Perturb a random subset of points
        num_points_to_move = rng.randint(1, 3)
        idxs = rng.sample(range(N), num_points_to_move)
        
        new_points = current_points.copy()
        for idx in idxs:
            noise = np.random.normal(0, step_size, 2)
            new_points[idx] += noise
        
        # Clamp to triangle
        new_points = _clamp_points(new_points)
        
        new_area = _min_triangle_area_fast(new_points)
        
        # Acceptance probability
        delta = new_area - current_area
        if delta > 0:
            # Always accept better solutions
            current_points = new_points
            current_area = new_area
            if current_area > best_area:
                best_area = current_area
                best_points = current_points.copy()
        else:
            # Accept worse solutions with probability exp(delta / T)
            if T > 1e-6:
                prob = np.exp(delta / T)
                if rng.random() < prob:
                    current_points = new_points
                    current_area = new_area
        
        # Cooling
        T *= alpha
        # Occasionally reheat or adjust step size if stuck? 
        # For now, simple cooling.
        
        iterations += 1
        # If T is very low, we are in local refinement mode.
        # We can reduce step size as we cool
        step_size = max(0.001, 0.05 * (T / 0.01))

    return best_points, best_area
=======
def _coordinate_descent(points, time_limit, rng):
    """
    Coordinate descent local search: move one point at a time to improve min area.
    This is more effective than SA for the non-smooth min-area objective.
    """
    start_time = time.time()
    current_points = points.copy()
    current_area = _min_triangle_area_fast(current_points)
    
    best_points = current_points.copy()
    best_area = current_area
    
    # Precompute index pairs for efficiency
    n = N
    all_indices = list(range(n))
    
    # Step sizes to try
    base_steps = [0.1, 0.05, 0.02, 0.01, 0.005, 0.002, 0.001]
    
    no_improve_count = 0
    max_no_improve = 500
    
    while time.time() < time_limit and no_improve_count < max_no_improve:
        improved = False
        
        # Try moving each point in random order
        for idx in rng.sample(all_indices, n):
            if time.time() >= time_limit:
                break
            
            # Try different step sizes
            for step in base_steps:
                # Try 8 directions
                for angle in np.linspace(0, 2 * np.pi, 8, endpoint=False):
                    dx = step * np.cos(angle)
                    dy = step * np.sin(angle)
                    
                    new_points = current_points.copy()
                    new_points[idx] = current_points[idx] + np.array([dx, dy])
                    new_points = _clamp_points(new_points)
                    
                    new_area = _min_triangle_area_fast(new_points)
                    
                    if new_area > current_area + 1e-12:
                        current_points = new_points
                        current_area = new_area
                        improved = True
                        no_improve_count = 0
                        
                        if current_area > best_area:
                            best_area = current_area
                            best_points = current_points.copy()
                        break  # Move to next point
                else:
                    continue
                break  # Move to next point
        
        if not improved:
            no_improve_count += 1
            # Reduce step sizes when stuck
            base_steps = [s * 0.5 for s in base_steps if s * 0.5 > 1e-5]
            if len(base_steps) == 0:
                break
    
    return best_points, best_area


def _simulated_annealing(points_init, time_limit, rng):
    """
    Run Simulated Annealing to maximize the min triangle area.
    """
    start_time = time.time()
    current_points = points_init.copy()
    current_area = _min_triangle_area_fast(current_points)
    
    best_points = current_points.copy()
    best_area = current_area
    
    # Initial temperature
    T = 0.01
    # Cooling rate
    alpha = 0.999
    # Step size
    step_size = 0.05
    
    iterations = 0
    while time.time() < time_limit:
        # Generate a new candidate
        # Perturb a random subset of points
        num_points_to_move = rng.randint(1, 3)
        idxs = rng.sample(range(N), num_points_to_move)
        
        new_points = current_points.copy()
        for idx in idxs:
            noise = np.random.normal(0, step_size, 2)
            new_points[idx] += noise
        
        # Clamp to triangle
        new_points = _clamp_points(new_points)
        
        new_area = _min_triangle_area_fast(new_points)
        
        # Acceptance probability
        delta = new_area - current_area
        if delta > 0:
            # Always accept better solutions
            current_points = new_points
            current_area = new_area
            if current_area > best_area:
                best_area = current_area
                best_points = current_points.copy()
        else:
            # Accept worse solutions with probability exp(delta / T)
            if T > 1e-6:
                prob = np.exp(delta / T)
                if rng.random() < prob:
                    current_points = new_points
                    current_area = new_area
        
        # Cooling
        T *= alpha
        # Occasionally reheat or adjust step size if stuck? 
        # For now, simple cooling.
        
        iterations += 1
        # If T is very low, we are in local refinement mode.
        # We can reduce step size as we cool
        step_size = max(0.001, 0.05 * (T / 0.01))

    return best_points, best_area
>>>>>>> REPLACE

