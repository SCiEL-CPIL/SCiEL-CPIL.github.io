<<<<<<< SEARCH
def _try_pack(outer_side, time_budget=5.0):
    """Try to find a valid packing. Returns (success, inner, max_violation)."""
    start_time = time.time()
    outer_center = (0.0, 0.0)
    outer_angle = 0.0
    
    # Start with hex grid
    inner = _make_hex_grid(N)
    
    # Try several relaxations with different initial perturbations
    best_inner = None
    best_viol = float('inf')
    
    np.random.seed(42)
    
    for restart in range(5):
        if time.time() - start_time > time_budget:
            break
        
        # Perturb initial configuration
        perturbed = []
        for i in range(N):
            x, y, a = inner[i]
            px = x + np.random.uniform(-0.5, 0.5)
            py = y + np.random.uniform(-0.5, 0.5)
            pa = np.random.uniform(0, 60)
            perturbed.append([px, py, pa])
        
        # Relax
        relaxed = _relax(perturbed, outer_side, outer_angle, outer_center, 
                        iterations=100, dt=0.05)
        
        valid, viol = _check_validity(relaxed, outer_side, outer_angle, outer_center)
        if viol < best_viol:
            best_viol = viol
            best_inner = relaxed
        
        if valid:
            return True, relaxed, viol
    
    return best_viol < 1e-6, best_inner, best_viol
=======
def _try_pack(outer_side, time_budget=5.0):
    """Try to find a valid packing. Returns (success, inner, max_violation)."""
    start_time = time.time()
    outer_center = (0.0, 0.0)
    outer_angle = 0.0
    
    # Start with hex grid
    base_inner = _make_hex_grid(N)
    
    best_inner = None
    best_viol = float('inf')
    
    np.random.seed(42)
    
    # Run multiple restarts
    while time.time() - start_time < time_budget:
        # Perturb initial configuration
        perturbed = []
        for i in range(N):
            x, y, a = base_inner[i]
            px = x + np.random.uniform(-0.2, 0.2)
            py = y + np.random.uniform(-0.2, 0.2)
            pa = np.random.uniform(-10, 10)
            perturbed.append([px, py, pa])
        
        # Optimize
        optimized = _optimize(perturbed, outer_side, outer_angle, outer_center, 
                              time_budget=min(1.0, time_budget - (time.time() - start_time)))
        
        valid, viol = _check_validity(optimized, outer_side, outer_angle, outer_center)
        if viol < best_viol:
            best_viol = viol
            best_inner = optimized
        
        if valid:
            return True, optimized, viol
            
    return best_viol < 1e-6, best_inner, best_viol
>>>>>>> REPLACE

