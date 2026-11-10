import numpy as np
import time

def moving_sofa_2d() -> np.ndarray:
    start_time = time.time()
    deadline = start_time + 580  # 50s margin before 630s kill

    # Grid definition
    xs = np.arange(-3.0, 1.5, 0.01)
    ys = np.arange(-1.5, 1.5, 0.01)
    X, Y = np.meshgrid(xs, ys, indexing='xy')
    X_flat = X.flatten()
    Y_flat = Y.flatten()
    n_points = X_flat.size

    P_start = np.array([0.0, 0.0, 0.0])
    P_end = np.array([-2.0, 0.0, 90.0])

    def densify_path(path):
        """Add linear interpolation between consecutive placements."""
        full_path = np.vstack([P_start, path, P_end])
        dense_points = [full_path[0]]
        for i in range(len(full_path) - 1):
            p1 = full_path[i]
            p2 = full_path[i+1]
            delta_a = abs(p2[2] - p1[2])
            m = max(1, int(np.ceil(delta_a)))
            for k in range(1, m + 1):
                t = k / m
                p_int = p1 + t * (p2 - p1)
                dense_points.append(p_int)
        return np.array(dense_points)

    def count_inside(placements):
        """Count grid points inside all placements using progressive pruning."""
        valid_indices = np.arange(n_points)
        
        for x, y, a_deg in placements:
            a_rad = np.deg2rad(a_deg)
            cos_a = np.cos(a_rad)
            sin_a = np.sin(a_rad)
            
            Xv = X_flat[valid_indices]
            Yv = Y_flat[valid_indices]
            
            dx = Xv - x
            dy = Yv - y
            
            u = dx * cos_a + dy * sin_a
            v = -dx * sin_a + dy * cos_a
            
            in_rect1 = (u >= -1e-9) & (u <= 1 + 1e-9) & (v >= -3 - 1e-9) & (v <= 1 + 1e-9)
            in_rect2 = (u >= -3 - 1e-9) & (u <= 1 + 1e-9) & (v >= -1e-9) & (v <= 1 + 1e-9)
            inside = in_rect1 | in_rect2
            
            if not np.any(inside):
                return 0
            
            valid_indices = valid_indices[inside]
            if len(valid_indices) == 0:
                return 0
        
        return len(valid_indices)

    def score_path(path):
        dense = densify_path(path)
        count = count_inside(dense)
        return count * 0.0001

    # Start with a reasonable initial path
    best_path = np.array([
        [0.0, 0.0, 5.0],
        [-0.1, 0.1, 10.0],
        [-0.2, 0.2, 20.0],
        [-0.3, 0.3, 30.0],
        [-0.4, 0.35, 45.0],
        [-0.5, 0.3, 60.0],
        [-0.6, 0.25, 70.0],
        [-0.7, 0.2, 80.0],
        [-0.8, 0.1, 85.0],
    ], dtype=np.float64)
    best_score = score_path(best_path)

    # Optimize x,y at fixed angles
    angles_opt = [5.0, 10.0, 20.0, 30.0, 45.0, 60.0, 70.0, 80.0, 85.0]
    n_wp = len(angles_opt)
    
    def path_from_params(params):
        path = np.zeros((n_wp, 3))
        for i, a in enumerate(angles_opt):
            path[i, 0] = params[2*i]
            path[i, 1] = params[2*i+1]
            path[i, 2] = a
        return path

    def objective(params):
        if time.time() > deadline:
            return 0
        path = path_from_params(params)
        return -score_path(path)

    # Bounds for x, y
    bounds = [(-1.5, 0.5), (-1.0, 2.0)] * n_wp

    # Timing a single evaluation
    t0 = time.time()
    test_params = np.array([-0.5, 0.5] * n_wp)
    score_path(path_from_params(test_params))
    t1 = time.time()
    avg_eval_time = t1 - t0
    
    time_left = deadline - time.time()
    max_evals = max(10, int(time_left * 0.7 / avg_eval_time))
    
    # Use L-BFGS-B with multiple restarts
    from scipy.optimize import minimize
    
    best_params = None
    best_lbfgs_score = 0
    
    # Generate multiple starting points
    np.random.seed(42)
    n_restarts = min(30, max(5, max_evals // 20))
    
    for restart in range(n_restarts):
        if time.time() > deadline - 30:
            break
        
        if restart == 0:
            # First restart: use the current best
            current_params = best_path[:, 0] * 0 + np.array([-0.5, 0.5] * n_wp)
        else:
            # Subsequent restarts: perturb the best or random start
            if best_params is not None and np.random.random() < 0.5:
                current_params = best_params + np.random.randn(len(best_params)) * 0.1
            else:
                current_params = np.random.uniform(
                    [b[0] for b in bounds], 
                    [b[1] for b in bounds]
                )
            current_params = np.clip(current_params, 
                                     [b[0] for b in bounds], 
                                     [b[1] for b in bounds])
        
        try:
            result = minimize(
                objective,
                current_params,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 200, 'ftol': 1e-10, 'gtol': 1e-8}
            )
            
            lbfgs_score = -result.fun
            if lbfgs_score > best_lbfgs_score:
                best_lbfgs_score = lbfgs_score
                best_params = result.x
                best_path_lbfgs = path_from_params(result.x)
                if lbfgs_score > best_score:
                    best_path = best_path_lbfgs
                    best_score = lbfgs_score
        except Exception:
            pass

    # Nelder-Mead refinement on the best L-BFGS-B result
    if time.time() < deadline - 60 and best_params is not None:
        try:
            nm_result = minimize(
                objective,
                best_params,
                method='Nelder-Mead',
                options={'maxiter': 500, 'xatol': 1e-5, 'fatol': 1e-7}
            )
            nm_score = -nm_result.fun
            if nm_score > best_score:
                best_params = nm_result.x
                best_path = path_from_params(nm_result.x)
                best_score = nm_score
        except Exception:
            pass

    # Multi-start local refinement
    if time.time() < deadline - 30 and best_params is not None:
        for trial in range(15):
            if time.time() > deadline - 15:
                break
            perturbation = np.random.randn(len(best_params)) * 0.05
            candidate = np.clip(best_params + perturbation, 
                                [b[0] for b in bounds], 
                                [b[1] for b in bounds])
            
            try:
                local_result = minimize(
                    objective,
                    candidate,
                    method='Nelder-Mead',
                    options={'maxiter': 100, 'xatol': 1e-5, 'fatol': 1e-7}
                )
                local_score = -local_result.fun
                if local_score > best_score:
                    best_params = local_result.x
                    best_path = path_from_params(local_result.x)
                    best_score = local_score
            except Exception:
                pass

    # Final coordinate ascent polish
    if time.time() < deadline - 10 and best_params is not None:
        current = best_params.copy()
        current_score = best_score
        step = 0.05
        improved = True
        while improved and step > 1e-6 and time.time() < deadline - 5:
            improved = False
            for i in range(len(current)):
                for direction in [-1, 1]:
                    new_val = current[i] + direction * step
                    if new_val < bounds[i][0] or new_val > bounds[i][1]:
                        continue
                    old_val = current[i]
                    current[i] = new_val
                    new_score = -objective(current)
                    if new_score > current_score:
                        current_score = new_score
                        improved = True
                    else:
                        current[i] = old_val
            step *= 0.5
            if improved:
                if current_score > best_score:
                    best_params = current.copy()
                    best_path = path_from_params(current)
                    best_score = current_score

    # Final validation
    final_score = score_path(best_path)
    if final_score > best_score:
        best_score = final_score
    
    return best_path
