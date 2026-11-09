<<<<<<< SEARCH
def _lloyd_optimize(points, time_limit):
    """
    Run alternating Lloyd's algorithm and projected gradient ascent
    with soft-max repulsion for a specified time limit.
    """
    start_time = time.time()
    
    # Create a fine grid for Voronoi approximation
    # 200x200 grid is ~40k points, very fast for N=130
    theta = np.linspace(0, 2*np.pi, 200)
    phi = np.linspace(0, np.pi, 200)
    TH, PH = np.meshgrid(theta, phi)
    X = np.sin(PH) * np.cos(TH)
    Y = np.sin(PH) * np.sin(TH)
    Z = np.cos(PH)
    grid = np.column_stack([X.ravel(), Y.ravel(), Z.ravel()])
    
    current = points.copy()
    best = current.copy()
    best_val = _covering_radius(best)
    
    # Adaptive learning rate: start larger, decay over time
    lr = 0.08
    lr_decay = 0.995
    
    for it in range(1000): # Max iterations, will break on time
        if time.time() - start_time > time_limit:
            break
        
        # Alternate between Lloyd's and gradient ascent
        if it % 3 == 0:
            # Lloyd's iteration (structural)
            new_pts = _lloyd_iteration(current, grid)
        else:
            # Projected gradient ascent with soft-max repulsion
            new_pts = _projected_gradient_step(current, lr=lr)
        
        # Check convergence
        if np.max(np.linalg.norm(new_pts - current, axis=1)) < 1e-9:
            break
            
        current = new_pts
        
        # Evaluate
        val = _covering_radius(current)
        if val < best_val:
            best_val = val
            best = current.copy()
        
        # Decay learning rate
        lr *= lr_decay
    
    return best, best_val
=======
def _lloyd_optimize(points, time_limit):
    """
    Run alternating Lloyd's algorithm, worst-facet shrinkage, and projected gradient ascent
    for a specified time limit.
    """
    start_time = time.time()
    
    # Create a fine grid for Voronoi approximation
    # 200x200 grid is ~40k points, very fast for N=130
    theta = np.linspace(0, 2*np.pi, 200)
    phi = np.linspace(0, np.pi, 200)
    TH, PH = np.meshgrid(theta, phi)
    X = np.sin(PH) * np.cos(TH)
    Y = np.sin(PH) * np.sin(TH)
    Z = np.cos(PH)
    grid = np.column_stack([X.ravel(), Y.ravel(), Z.ravel()])
    
    current = points.copy()
    best = current.copy()
    best_val, _, _ = _covering_radius_with_worst_facet(best)
    
    # Adaptive learning rate: start larger, decay over time
    lr = 0.08
    lr_decay = 0.995
    
    # Track consecutive non-improvements for adaptive strategy
    non_improve_count = 0
    
    for it in range(1000): # Max iterations, will break on time
        if time.time() - start_time > time_limit:
            break
        
        # Choose strategy based on iteration and convergence
        if it % 4 == 0:
            # Lloyd's iteration (structural)
            new_pts = _lloyd_iteration(current, grid)
        elif it % 4 == 1:
            # Worst facet shrinkage (direct geometric correction)
            val, worst_facet, worst_cc = _covering_radius_with_worst_facet(current)
            new_pts = _shrink_worst_facet(current, worst_facet, worst_cc, step_size=0.04)
        elif it % 4 == 2:
            # Worst-facet-weighted gradient step
            new_pts = _worst_facet_gradient_step(current, lr=lr)
        else:
            # Standard projected gradient ascent
            new_pts = _projected_gradient_step(current, lr=lr)
        
        # Check convergence
        if np.max(np.linalg.norm(new_pts - current, axis=1)) < 1e-9:
            break
            
        current = new_pts
        
        # Evaluate
        val, _, _ = _covering_radius_with_worst_facet(current)
        if val < best_val:
            best_val = val
            best = current.copy()
            non_improve_count = 0
        else:
            non_improve_count += 1
        
        # Decay learning rate
        lr *= lr_decay
        
        # If stuck for many iterations, try a larger step
        if non_improve_count > 20:
            lr = max(lr * 1.5, 0.02)
            non_improve_count = 0
    
    return best, best_val
>>>>>>> REPLACE

