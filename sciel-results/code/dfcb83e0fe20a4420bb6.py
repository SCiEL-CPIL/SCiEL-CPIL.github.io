from scipy.optimize import minimize
import numpy as np
import time
from scipy.optimize import differential_evolution

def moving_sofa_2d() -> np.ndarray:
    start_time = time.time()
    deadline = start_time + 580  # 20s margin

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

    def check_inside_fast(placements):
        # Adaptive mask
        valid_mask = np.ones(n_points, dtype=bool)
        valid_indices = np.arange(n_points)
        
        for x, y, a_deg in placements:
            a_rad = np.deg2rad(a_deg)
            cos_a = np.cos(a_rad)
            sin_a = np.sin(a_rad)
            
            # Only check valid points
            Xv = X_flat[valid_indices]
            Yv = Y_flat[valid_indices]
            
            dx = Xv - x
            dy = Yv - y
            
            u = dx * cos_a + dy * sin_a
            v = -dx * sin_a + dy * cos_a
            
            in_rect1 = (u >= 0) & (u <= 1) & (v >= -3) & (v <= 1)
            in_rect2 = (u >= -3) & (u <= 1) & (v >= 0) & (v <= 1)
            inside = in_rect1 | in_rect2
            
            if not np.any(inside):
                return 0
            
            # Update valid indices
            valid_indices = valid_indices[inside]
            
        return len(valid_indices)

    def score_path(path):
        dense = densify_path(path)
        count = check_inside_fast(dense)
        return count * 0.0001

    # Better initial path with 7 waypoints
    angles_opt = [10.0, 20.0, 30.0, 45.0, 60.0, 70.0, 80.0]
    n_waypoints = len(angles_opt)
    
    def path_from_params(params):
        # params: [x1, y1, x2, y2, ..., x7, y7]
        path = np.zeros((n_waypoints, 3))
        for i, a in enumerate(angles_opt):
            path[i, 0] = params[2*i]
            path[i, 1] = params[2*i+1]
            path[i, 2] = a
        return path

    def objective(params):
        if time.time() > deadline:
            return 0
        path = path_from_params(params)
        return -score_path(path) # Minimize negative score

    # Better initial path
    best_path = np.array([
        [-0.05, 0.05, 10.0],
        [-0.10, 0.10, 20.0],
        [-0.20, 0.15, 30.0],
        [-0.30, 0.20, 45.0],
        [-0.45, 0.15, 60.0],
        [-0.55, 0.10, 70.0],
        [-0.65, 0.15, 80.0],
    ], dtype=np.float64)
    best_score = score_path(best_path)
    best_params = best_path[:, :2].flatten()

    # Bounds for x, y - wider range for exploration
    bounds = [(-1.8, 0.8), (-0.8, 1.8)] * n_waypoints
    
    # Run Differential Evolution
    # maxiter and popsize tuned for time budget
    # Let's start with a small pop and few iterations, then expand if time permits.
    
    # Run Differential Evolution
    t0 = time.time()
    # Quick test to estimate evaluation time
    test_params = np.random.default_rng(42).uniform([-1.8, -0.8]*n_waypoints, [0.8, 1.8]*n_waypoints)
    score_path(path_from_params(test_params))
    t1 = time.time()
    avg_eval_time = t1 - t0
    
    time_left = deadline - time.time()
    # Estimate max evaluations
    max_evals = int(time_left / avg_eval_time)
    # DE overhead is roughly 2-3x the number of function evaluations
    max_iter = max(15, min(100, max_evals // 80)) 
    popsize = 25
    
    print(f"Starting DE with max_iter={max_iter}, popsize={popsize}, time_left={time_left:.1f}s")
    
    try:
        result = differential_evolution(
            objective, 
            bounds, 
            maxiter=max_iter, 
            popsize=popsize,
            seed=42,
            polish=False,
            init='sobol',
            tol=0.01,
            update_interval=1
        )
        
        if result.fun < 0: # If we found a better path
            best_params_de = result.x
            best_path_de = path_from_params(best_params_de)
            best_score_de = -result.fun
            if best_score_de > best_score:
                best_path = best_path_de
                best_score = best_score_de
                best_params = best_params_de
            print(f"DE result: score={best_score_de:.6f}")
    except Exception as e:
        print(f"DE failed: {e}")

    # Local refinement using Nelder-Mead
    if time.time() < deadline - 60:
        print("Starting Nelder-Mead refinement")
        current_params = best_params.copy()
        
        try:
            nm_result = minimize(
                objective,
                current_params,
                method='Nelder-Mead',
                bounds=bounds,
                options={
                    'maxiter': 200,
                    'maxfev': 500,
                    'xatol': 0.005,
                    'fatol': 0.0001,
                    'adaptive': True
                }
            )
            
            if nm_result.fun < 0:
                best_params_nm = nm_result.x
                best_path_nm = path_from_params(best_params_nm)
                best_score_nm = -nm_result.fun
                if best_score_nm > best_score:
                    best_path = best_path_nm
                    best_score = best_score_nm
                    best_params = best_params_nm
                print(f"NM result: score={best_score_nm:.6f}")
        except Exception as e:
            print(f"NM failed: {e}")

    # Additional local refinement with smaller steps
    if time.time() < deadline - 30:
        print("Starting fine-tuning")
        current_params = best_params.copy()
        current_score = -objective(current_params)
        
        rng = np.random.default_rng(123)
        step_sizes = [0.05, 0.02, 0.01, 0.005]
        
        for step_size in step_sizes:
            if time.time() > deadline - 10:
                break
            
            improved = True
            while improved and time.time() < deadline - 10:
                improved = False
                for _ in range(50):
                    if time.time() > deadline - 10:
                        break
                    perturbation = rng.normal(0, step_size, size=current_params.shape)
                    new_params = current_params + perturbation
                    new_params[0::2] = np.clip(new_params[0::2], -1.8, 0.8)
                    new_params[1::2] = np.clip(new_params[1::2], -0.8, 1.8)
                    
                    new_score = -objective(new_params)
                    if new_score > current_score:
                        current_params = new_params
                        current_score = new_score
                        improved = True
                        if current_score > best_score:
                            best_path = path_from_params(current_params)
                            best_score = current_score

    final_path = path_from_params(best_params)
    final_score = score_path(final_path)
    if final_score > best_score:
        best_path = final_path
        best_score = final_score

    print(f"Final score: {best_score:.6f}")
    return best_path
