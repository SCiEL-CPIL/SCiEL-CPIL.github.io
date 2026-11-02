import numpy as np
import time
import scipy.optimize as opt

N = 13

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    time_limit = 140.0  # Leave 10s margin for safety
    
    # Precompute indices for all triples
    triples = [(i, j, k) for i in range(N) for j in range(i+1, N) for k in range(j+1, N)]
    n_triples = len(triples)
    
    def compute_score(points):
        """Compute min triangle area / convex hull area for given points."""
        # Compute convex hull area
        # We'll use scipy.spatial.ConvexHull
        from scipy.spatial import ConvexHull
        try:
            hull = ConvexHull(points)
            hull_area = hull.volume  # In 2D, volume is area
        except Exception:
            # Fallback: shoelace formula on sorted points (assumes convex position)
            angles = np.arctan2(points[:, 1], points[:, 0])
            sorted_idx = np.argsort(angles)
            p_sorted = points[sorted_idx]
            x = p_sorted[:, 0]
            y = p_sorted[:, 1]
            x_next = np.roll(x, -1)
            y_next = np.roll(y, -1)
            hull_area = 0.5 * np.abs(np.sum(x * y_next - x_next * y))
        
        if hull_area < 1e-9:
            return 0.0, 0.0
            
        # Triangle areas
        min_area = float('inf')
        for i, j, k in triples:
            v1x = points[j, 0] - points[i, 0]
            v1y = points[j, 1] - points[i, 1]
            v2x = points[k, 0] - points[i, 0]
            v2y = points[k, 1] - points[i, 1]
            cross = abs(v1x * v2y - v1y * v2x)
            area = 0.5 * cross
            if area < min_area:
                min_area = area
                
        score = min_area / hull_area
        return score, hull_area

    def objective_from_points(points):
        score, _ = compute_score(points)
        return -score

    # Strategy: Use concentric regular polygons as initial configurations
    # Then optimize with L-BFGS-B
    
    def make_concentric_polygons(n_outer, n_inner, radius_inner=0.5, phase_outer=0, phase_inner=0):
        """Create points on two concentric circles."""
        points = []
        # Outer circle
        if n_outer > 0:
            theta_outer = np.linspace(phase_outer, 2*np.pi + phase_outer, n_outer, endpoint=False)
            x_outer = np.cos(theta_outer)
            y_outer = np.sin(theta_outer)
            points.extend(zip(x_outer, y_outer))
        # Inner circle
        if n_inner > 0:
            theta_inner = np.linspace(phase_inner, 2*np.pi + phase_inner, n_inner, endpoint=False)
            x_inner = radius_inner * np.cos(theta_inner)
            y_inner = radius_inner * np.sin(theta_inner)
            points.extend(zip(x_inner, y_inner))
        return np.array(points)

    def make_single_polygon(n, radius=1.0, phase=0):
        """Create points on a single circle."""
        theta = np.linspace(phase, 2*np.pi + phase, n, endpoint=False)
        return np.column_stack((radius * np.cos(theta), radius * np.sin(theta)))

    # Generate diverse initial configurations
    initial_configs = []
    
    # 1. Regular 13-gon
    initial_configs.append(make_single_polygon(13))
    
    # 2. Concentric: 7 outer + 6 inner
    initial_configs.append(make_concentric_polygons(7, 6, radius_inner=0.5))
    
    # 3. Concentric: 6 outer + 7 inner
    initial_configs.append(make_concentric_polygons(6, 7, radius_inner=0.5))
    
    # 4. Concentric with different phase
    initial_configs.append(make_concentric_polygons(7, 6, radius_inner=0.4, phase_inner=np.pi/6))
    
    # 5. Concentric with different radius
    initial_configs.append(make_concentric_polygons(7, 6, radius_inner=0.6))
    
    # 6. Concentric: 8 outer + 5 inner
    initial_configs.append(make_concentric_polygons(8, 5, radius_inner=0.5))
    
    # 7. Concentric: 5 outer + 8 inner
    initial_configs.append(make_concentric_polygons(5, 8, radius_inner=0.5))
    
    # 8. Regular 13-gon with slight perturbation
    theta_pert = np.linspace(0, 2*np.pi, 13, endpoint=False) + np.random.uniform(-0.05, 0.05, 13)
    initial_configs.append(np.column_stack((np.cos(theta_pert), np.sin(theta_pert))))
    
    # 9. Another concentric configuration
    initial_configs.append(make_concentric_polygons(7, 6, radius_inner=0.3, phase_outer=np.pi/14, phase_inner=np.pi/6))
    
    # 10. Random-ish but structured
    np.random.seed(42)
    for _ in range(3):
        n_outer = np.random.randint(5, 9)
        n_inner = 13 - n_outer
        r = np.random.uniform(0.3, 0.7)
        phase_o = np.random.uniform(0, 2*np.pi)
        phase_i = np.random.uniform(0, 2*np.pi)
        initial_configs.append(make_concentric_polygons(n_outer, n_inner, radius_inner=r, phase_outer=phase_o, phase_inner=phase_i))

    # Evaluate all initial configurations
    best_score = -np.inf
    best_points = None
    best_config_idx = -1
    
    for idx, config in enumerate(initial_configs):
        if time.time() - start_time > time_limit - 20:
            break
        score, _ = compute_score(config)
        if score > best_score:
            best_score = score
            best_points = config.copy()
            best_config_idx = idx

    # Now optimize from the best configuration using L-BFGS-B
    def objective_flat(x_flat):
        points = x_flat.reshape(-1, 2)
        return objective_from_points(points)

    if best_points is not None and time.time() - start_time < time_limit - 15:
        x0 = best_points.flatten()
        
        # Use L-BFGS-B with bounds to keep points in a reasonable range
        bounds = [(-1.5, 1.5)] * (2 * N)
        
        stop_flag = False
        def callback(xk):
            nonlocal stop_flag
            if time.time() - start_time > time_limit - 5:
                stop_flag = True
                return True
            return False

        try:
            res = opt.minimize(objective_flat, x0, method='L-BFGS-B', 
                              bounds=bounds,
                              options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-8},
                              callback=callback)
            
            if not stop_flag:
                res_points = res.x.reshape(-1, 2)
                score_res, _ = compute_score(res_points)
                if score_res > best_score:
                    best_score = score_res
                    best_points = res_points
        except Exception as e:
            pass

    # If time remains, try optimizing from other good initial configs
    if time.time() - start_time < time_limit - 20:
        # Sort configs by their initial score
        config_scores = []
        for idx, config in enumerate(initial_configs):
            if idx == best_config_idx:
                continue
            score, _ = compute_score(config)
            config_scores.append((score, idx))
        
        config_scores.sort(reverse=True)
        
        for score_init, idx in config_scores[:3]:  # Try top 3 other configs
            if time.time() - start_time > time_limit - 10:
                break
            
            x0 = initial_configs[idx].flatten()
            bounds = [(-1.5, 1.5)] * (2 * N)
            
            stop_flag = False
            def callback(xk):
                nonlocal stop_flag
                if time.time() - start_time > time_limit - 5:
                    stop_flag = True
                    return True
                return False

            try:
                res = opt.minimize(objective_flat, x0, method='L-BFGS-B', 
                                  bounds=bounds,
                                  options={'maxiter': 100, 'ftol': 1e-12, 'gtol': 1e-8},
                                  callback=callback)
                
                if not stop_flag:
                    res_points = res.x.reshape(-1, 2)
                    score_res, _ = compute_score(res_points)
                    if score_res > best_score:
                        best_score = score_res
                        best_points = res_points
            except Exception:
                pass

    # Final polish with Nelder-Mead if time remains
    if best_points is not None and time.time() - start_time < time_limit - 10:
        def objective_points(points):
            score, _ = compute_score(points)
            return -score
        
        stop_flag = False
        def callback(xk):
            nonlocal stop_flag
            if time.time() - start_time > time_limit - 3:
                stop_flag = True
                return True
            return False

        try:
            # Nelder-Mead on flattened points
            res = opt.minimize(objective_flat, best_points.flatten(), method='Nelder-Mead',
                              options={'maxiter': 500, 'adaptive': True, 'xatol': 1e-10, 'fatol': 1e-10},
                              callback=callback)
            
            if not stop_flag:
                res_points = res.x.reshape(-1, 2)
                score_res, _ = compute_score(res_points)
                if score_res > best_score:
                    best_score = score_res
                    best_points = res_points
        except Exception:
            pass

    # Ensure we have valid points
    if best_points is None:
        best_points = make_single_polygon(13)

    return best_points

# EVOLVE-BLOCK-END
