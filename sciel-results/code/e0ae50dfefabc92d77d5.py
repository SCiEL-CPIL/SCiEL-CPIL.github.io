<<<<<<< SEARCH
def construct():
    start_time = time.time()
    timeout = 115
    
    best_c = None
    best_val = -np.inf
    
    inits = []
    
    # 1. Grids
    for r, c in [(6,6), (5,7), (7,5), (4,8), (8,4), (6,5), (5,6)]:
        inits.append(create_grid_init(r, c, N))
        
    # 2. Halton
    inits.append(halton_sequence(N).flatten())
    
    # 3. Phyllotaxis
    inits.append(phyllotaxis_points(N).flatten())
    
    # 4. Jittered Grids
    np.random.seed(42)
    base = create_grid_init(6, 6, N).reshape(N, 2)
    for jitter_scale in [0.02, 0.04, 0.06]:
        for _ in range(5):
            jitter = np.random.uniform(-jitter_scale, jitter_scale, size=(N, 2))
            p = np.clip(base + jitter, 0.01, 0.99)
            inits.append(p.flatten())
            
    # 5. Random
    for _ in range(10):
        p = np.random.uniform(0.05, 0.95, size=(N, 2))
        inits.append(p.flatten())
        
    bounds = [(0.001, 0.999), (0.001, 0.999)] * N
    
    for idx, c_init in enumerate(inits):
        if time.time() - start_time > timeout - 20:
            break
            
        # 1. L-BFGS-B
        try:
            res = minimize(lambda c: -sum_radii(c), c_init, jac=lambda c: -grad_sum_radii(c),
                           method='L-BFGS-B', bounds=bounds,
                           options={'maxiter': 300, 'ftol': 1e-14, 'gtol': 1e-12})
            val = -res.fun
            if val > best_val:
                best_val = val
                best_c = res.x.copy()
        except:
            pass
        
        # 2. Coordinate Descent Refinement
        if time.time() - start_time < timeout - 10:
            try:
                # Refine the result of LBFGS
                refined = coordinate_descent_refine(res.x if 'res' in locals() and res.x is not None else c_init, max_sweeps=5, time_limit=5.0)
                val = sum_radii(refined)
                if val > best_val:
                    best_val = val
                    best_c = refined.copy()
            except:
                pass
        
        if time.time() - start_time > timeout - 15:
            break

    if best_c is not None:
        # Final Polish
        if time.time() - start_time < timeout - 5:
            try:
                res = minimize(lambda c: -sum_radii(c), best_c, jac=lambda c: -grad_sum_radii(c),
                               method='L-BFGS-B', bounds=bounds,
                               options={'maxiter': 200, 'ftol': 1e-14, 'gtol': 1e-12})
                val = -res.fun
                if val > best_val:
                    best_val = val
                    best_c = res.x.copy()
            except:
                pass
        
        if time.time() - start_time < timeout - 2:
            try:
                refined = coordinate_descent_refine(best_c, max_sweeps=5, time_limit=3.0)
                val = sum_radii(refined)
                if val > best_val:
                    best_val = val
                    best_c = refined.copy()
            except:
                pass

    if best_c is None:
        best_c = create_grid_init(6, 6, N)
        
    p = best_c.reshape(N, 2)
    diff = p[:, np.newaxis, :] - p[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    np.fill_diagonal(dists, np.inf)
    min_neighbor_dist = np.min(dists, axis=1) / 2.0
    dist_walls = np.minimum(p, 1 - p)
    min_wall_dist = np.min(dist_walls, axis=1)
    radii = np.minimum(min_wall_dist, min_neighbor_dist)
    
    result = np.column_stack((p, radii))
    return result
=======
def construct():
    start_time = time.time()
    timeout = 115
    
    best_c = None
    best_val = -np.inf
    
    inits = []
    
    # 1. Grids
    for r, c in [(6,6), (5,7), (7,5), (4,8), (8,4), (6,5), (5,6)]:
        inits.append(create_grid_init(r, c, N))
        
    # 2. Halton
    inits.append(halton_sequence(N).flatten())
    
    # 3. Phyllotaxis
    inits.append(phyllotaxis_points(N).flatten())
    
    # 4. Jittered Grids
    np.random.seed(42)
    base = create_grid_init(6, 6, N).reshape(N, 2)
    for jitter_scale in [0.02, 0.04, 0.06]:
        for _ in range(5):
            jitter = np.random.uniform(-jitter_scale, jitter_scale, size=(N, 2))
            p = np.clip(base + jitter, 0.01, 0.99)
            inits.append(p.flatten())
            
    # 5. Random
    for _ in range(10):
        p = np.random.uniform(0.05, 0.95, size=(N, 2))
        inits.append(p.flatten())
        
    bounds = [(0.001, 0.999), (0.001, 0.999)] * N
    
    for idx, c_init in enumerate(inits):
        if time.time() - start_time > timeout - 10:
            break
            
        # Apply repulsion relaxation to get a better start
        p_relaxed = repulsion_relax(c_init.reshape(N, 2), iterations=10, step_size=0.01)
        
        # 1. L-BFGS-B with Soft Gradient
        try:
            res = minimize(lambda c: -soft_sum_radii(c), p_relaxed.flatten(), jac=lambda c: -grad_soft_sum_radii(c),
                           method='L-BFGS-B', bounds=bounds,
                           options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-10})
            val = sum_radii(res.x)
            if val > best_val:
                best_val = val
                best_c = res.x.copy()
        except:
            pass
        
        # 2. Coordinate Descent Refinement (Short)
        if time.time() - start_time < timeout - 5 and 'res' in locals() and res.x is not None:
            try:
                refined = coordinate_descent_refine(res.x, max_sweeps=2, time_limit=2.0)
                val = sum_radii(refined)
                if val > best_val:
                    best_val = val
                    best_c = refined.copy()
            except:
                pass

    if best_c is not None:
        # Final Polish with Hard Gradient (for precision)
        if time.time() - start_time < timeout - 2:
            try:
                res = minimize(lambda c: -sum_radii(c), best_c, jac=lambda c: -grad_sum_radii(c),
                               method='L-BFGS-B', bounds=bounds,
                               options={'maxiter': 100, 'ftol': 1e-14, 'gtol': 1e-12})
                val = sum_radii(res.x)
                if val > best_val:
                    best_val = val
                    best_c = res.x.copy()
            except:
                pass

    if best_c is None:
        best_c = create_grid_init(6, 6, N)
        
    p = best_c.reshape(N, 2)
    diff = p[:, np.newaxis, :] - p[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    np.fill_diagonal(dists, np.inf)
    min_neighbor_dist = np.min(dists, axis=1) / 2.0
    dist_walls = np.minimum(p, 1 - p)
    min_wall_dist = np.min(dist_walls, axis=1)
    radii = np.minimum(min_wall_dist, min_neighbor_dist)
    
    result = np.column_stack((p, radii))
    return result
>>>>>>> REPLACE
