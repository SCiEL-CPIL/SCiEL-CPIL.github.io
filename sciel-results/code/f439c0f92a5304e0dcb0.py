<<<<<<< SEARCH
# EVOLVE-BLOCK-START
def construct():
    """
    Constructs 13 points to maximize the minimum triangle area relative to convex hull area.
    Starts from regular 13-gon and uses gradient-based optimization with smooth surrogate,
    then polishes with exact local optimization.
    """
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s limit
    
    # Initialize with regular 13-gon
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
    points = np.column_stack((np.cos(angles), np.sin(angles)))
    
    best_points = points.copy()
    best_score = _min_triangle_area_ratio(points)
    
    # Phase 1: Coarse optimization with larger temperature (smoother landscape)
    temperatures = [0.1, 0.05, 0.02, 0.01, 0.005]
    
    for temp in temperatures:
        if time.time() > deadline:
            break
        
        x0 = points.flatten()
        
        try:
            result = minimize(
                _negative_soft_min_ratio,
                x0,
                args=(temp,),
                method='L-BFGS-B',
                options={'maxiter': 200, 'ftol': 1e-10, 'gtol': 1e-8}
            )
            
            if result.success or result.fun < -best_score:
                new_points = result.x.reshape(N, 2)
                new_score = _min_triangle_area_ratio(new_points)
                
                if new_score > best_score:
                    best_points = new_points.copy()
                    best_score = new_score
                    points = new_points.copy()
        except Exception:
            pass
    
    # Phase 2: Fine-tune with exact objective using local search
    if time.time() < deadline:
        x0 = best_points.flatten()
        
        try:
            result = minimize(
                _negative_exact_ratio,
                x0,
                method='Nelder-Mead',
                options={'maxiter': 500, 'xatol': 1e-10, 'fatol': 1e-12, 'adaptive': True}
            )
            
            if result.fun < -best_score:
                new_points = result.x.reshape(N, 2)
                new_score = _min_triangle_area_ratio(new_points)
                
                if new_score > best_score:
                    best_points = new_points.copy()
                    best_score = new_score
        except Exception:
            pass
    
    # Phase 3: Final polish with very tight tolerance
    if time.time() < deadline - 5:
        x0 = best_points.flatten()
        
        try:
            result = minimize(
                _negative_exact_ratio,
                x0,
                method='Nelder-Mead',
                options={'maxiter': 200, 'xatol': 1e-12, 'fatol': 1e-14, 'adaptive': True}
            )
            
            if result.fun < -best_score:
                new_points = result.x.reshape(N, 2)
                new_score = _min_triangle_area_ratio(new_points)
                
                if new_score > best_score:
                    best_points = new_points.copy()
                    best_score = new_score
        except Exception:
            pass
    
    return best_points
# EVOLVE-BLOCK-END
=======
# EVOLVE-BLOCK-START
def construct():
    """
    Constructs 13 points to maximize the minimum triangle area relative to convex hull area.
    Uses multi-start with gradient-based optimization and local polish.
    """
    start_time = time.time()
    deadline = start_time + 145  # Safety margin before 150s limit
    
    def time_remaining():
        return deadline - time.time()
    
    best_points = None
    best_score = 0.0
    
    def update_best(points):
        nonlocal best_points, best_score
        score = _min_triangle_area_ratio(points)
        if score > best_score:
            best_score = score
            best_points = points.copy()
            return True
        return False
    
    # Multiple initial configurations
    initial_configs = []
    
    # 1. Regular 13-gon
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
    initial_configs.append(np.column_stack((np.cos(angles), np.sin(angles))))
    
    # 2. Regular 13-gon with slight perturbation
    np.random.seed(42)
    perturbed = initial_configs[0].copy()
    perturbed += np.random.randn(N, 2) * 0.05
    initial_configs.append(perturbed)
    
    # 3. Two concentric rings (7 outer, 6 inner)
    outer_angles = np.linspace(0, 2 * np.pi, 7, endpoint=False)
    inner_angles = outer_angles + np.pi / 7
    outer = np.column_stack((np.cos(outer_angles), np.sin(outer_angles)))
    inner = np.column_stack((0.5 * np.cos(inner_angles), 0.5 * np.sin(inner_angles)))
    initial_configs.append(np.vstack([outer, inner]))
    
    # 4. Three rings (5-4-4)
    ring1_angles = np.linspace(0, 2 * np.pi, 5, endpoint=False)
    ring2_angles = ring1_angles + np.pi / 5
    ring3_angles = ring1_angles
    r1 = np.column_stack((np.cos(ring1_angles), np.sin(ring1_angles)))
    r2 = np.column_stack((0.6 * np.cos(ring2_angles), 0.6 * np.sin(ring2_angles)))
    r3 = np.column_stack((0.3 * np.cos(ring3_angles), 0.3 * np.sin(ring3_angles)))
    initial_configs.append(np.vstack([r1, r2, r3]))
    
    # 5. Random points in unit disk
    np.random.seed(123)
    theta = np.random.uniform(0, 2 * np.pi, N)
    r = np.sqrt(np.random.uniform(0, 1, N))
    initial_configs.append(np.column_stack((r * np.cos(theta), r * np.sin(theta))))
    
    # Process each initial configuration
    for init_idx, init_points in enumerate(initial_configs):
        if time_remaining() < 10:
            break
        
        update_best(init_points)
        
        # Phase 1: Coarse optimization with soft-min
        temperatures = [0.1, 0.05, 0.02, 0.01, 0.005]
        
        for temp in temperatures:
            if time_remaining() < 5:
                break
            
            x0 = best_points.flatten() if best_points is not None else init_points.flatten()
            
            try:
                result = minimize(
                    _negative_soft_min_ratio,
                    x0,
                    args=(temp,),
                    method='L-BFGS-B',
                    options={'maxiter': 100, 'ftol': 1e-10, 'gtol': 1e-8}
                )
                
                if result.success or result.fun < -best_score:
                    new_points = result.x.reshape(N, 2)
                    update_best(new_points)
            except Exception:
                pass
        
        # Phase 2: Fine-tune with exact objective using Nelder-Mead
        if time_remaining() < 3:
            break
        
        x0 = best_points.flatten()
        
        try:
            result = minimize(
                _negative_exact_ratio,
                x0,
                method='Nelder-Mead',
                options={'maxiter': 300, 'xatol': 1e-10, 'fatol': 1e-12, 'adaptive': True}
            )
            
            if result.fun < -best_score:
                new_points = result.x.reshape(N, 2)
                update_best(new_points)
        except Exception:
            pass
        
        # Phase 3: Final high-precision polish
        if time_remaining() < 2:
            break
        
        x0 = best_points.flatten()
        
        try:
            result = minimize(
                _negative_exact_ratio,
                x0,
                method='Nelder-Mead',
                options={'maxiter': 200, 'xatol': 1e-12, 'fatol': 1e-14, 'adaptive': True}
            )
            
            if result.fun < -best_score:
                new_points = result.x.reshape(N, 2)
                update_best(new_points)
        except Exception:
            pass
    
    # If we still have time, do additional local search from best point
    if best_points is not None and time_remaining() > 5:
        x0 = best_points.flatten()
        
        try:
            result = minimize(
                _negative_exact_ratio,
                x0,
                method='Nelder-Mead',
                options={'maxiter': 500, 'xatol': 1e-13, 'fatol': 1e-15, 'adaptive': True}
            )
            
            if result.fun < -best_score:
                new_points = result.x.reshape(N, 2)
                update_best(new_points)
        except Exception:
            pass
    
    return best_points
# EVOLVE-BLOCK-END
>>>>>>> REPLACE
