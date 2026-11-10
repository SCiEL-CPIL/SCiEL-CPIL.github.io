# EVOLVE-BLOCK-START
import numpy as np
import time


def _build_grid(subsample=1):
    """
    Precompute grid points for sofa membership testing.
    subsample: 1 for full grid, >1 for subsampled grid (for optimization speed).
    """
    xs = np.arange(-3.0, 1.5, 0.01 * subsample)
    ys = np.arange(-1.5, 1.5, 0.01 * subsample)
    X, Y = np.meshgrid(xs, ys, indexing='ij')
    return X.ravel(), Y.ravel()


def _densify_path(path):
    """
    Densify a path the same way the evaluator does.
    Prepends (0,0,0) and appends (-2,0,90).
    Between consecutive rows, adds linear interpolations in m = max(1, ceil(|a2-a1|)) steps.
    """
    # Prepend start, append end
    full = np.vstack([[0.0, 0.0, 0.0], path, [-2.0, 0.0, 90.0]])
    
    # Densify
    result = [full[0]]
    for i in range(len(full) - 1):
        p1 = full[i]
        p2 = full[i + 1]
        da = abs(p2[2] - p1[2])
        m = max(1, int(np.ceil(da)))
        for k in range(1, m + 1):
            t = k / m
            interp = p1 + t * (p2 - p1)
            result.append(interp)
    return np.array(result)


def _check_in_base(u, v):
    """
    Check if point (u, v) is in the base L-shaped corridor.
    Base: {0 <= u <= 1, -3 <= v <= 1} ∪ {-3 <= u <= 1, 0 <= v <= 1}
    Returns boolean array.
    """
    # Arm 1: 0 <= u <= 1, -3 <= v <= 1
    arm1 = (u >= 0) & (u <= 1) & (v >= -3) & (v <= 1)
    # Arm 2: -3 <= u <= 1, 0 <= v <= 1
    arm2 = (u >= -3) & (u <= 1) & (v >= 0) & (v <= 1)
    return arm1 | arm2


def _evaluate_path(path, grid_x, grid_y):
    """
    Evaluate the sofa area for a given path.
    Returns the raw value (area).
    """
    densified = _densify_path(path)
    n_points = len(grid_x)
    
    # Start with all points as potential sofa members
    inside_all = np.ones(n_points, dtype=bool)
    
    for (cx, cy, a_deg) in densified:
        a_rad = np.radians(a_deg)
        cos_a = np.cos(a_rad)
        sin_a = np.sin(a_rad)
        
        # R(-a) * (w - c)
        dx = grid_x - cx
        dy = grid_y - cy
        
        # R(-a) = [[cos(a), sin(a)], [-sin(a), cos(a)]]
        u = dx * cos_a + dy * sin_a
        v = -dx * sin_a + dy * cos_a
        
        in_base = _check_in_base(u, v)
        inside_all &= in_base
        
        # Early exit if no points remain
        if not np.any(inside_all):
            return 0.0
    
    return float(np.sum(inside_all)) * 0.0001


def moving_sofa_2d() -> np.ndarray:
    """
    Optimize the moving sofa path to maximize the sofa area.
    Uses subsampled grid for fast iteration, then validates with full grid.
    """
    start_time = time.time()
    time_limit = 580  # Leave margin for safety (limit is 630)
    
    # Build grids - use subsampled for optimization, full for final evaluation
    grid_x_full, grid_y_full = _build_grid(subsample=1)
    grid_x_sub, grid_y_sub = _build_grid(subsample=3)
    
    # Initial path
    best_path = np.array([
        [0.0, 0.0, 10.0],
        [-0.2, 0.2, 20.0],
        [-0.5, 0.5, 30.0],
        [-0.8, 0.8, 45.0],
        [-1.2, 0.8, 60.0],
        [-1.6, 0.5, 75.0],
        [-1.9, 0.2, 85.0]
    ], dtype=np.float64)
    
    # Evaluate on subsampled grid for speed
    best_score_sub = _evaluate_path(best_path, grid_x_sub, grid_y_sub)
    best_score_full = _evaluate_path(best_path, grid_x_full, grid_y_full)
    
    n_params = best_path.size
    bounds = [(-2.0, 1.0), (-1.0, 1.0), (0.0, 90.0)] * 7
    
    current_params = best_path.ravel().copy()
    current_score_sub = best_score_sub
    
    step_sizes = [0.1, 0.1, 0.5] * 7
    
    def eval_sub(params):
        path = params.reshape(7, 3)
        for i in range(1, 7):
            if path[i, 2] < path[i-1, 2]:
                path[i, 2] = path[i-1, 2]
        return _evaluate_path(path, grid_x_sub, grid_y_sub)
    
    # Coordinate descent on subsampled grid
    max_iterations = 10000
    for i_iter in range(max_iterations):
        if time.time() - start_time > time_limit - 30:
            break
        
        improved = False
        for i in range(n_params):
            for sign in [1, -1]:
                new_params = current_params.copy()
                new_params[i] += sign * step_sizes[i]
                
                b_min, b_max = bounds[i]
                if new_params[i] < b_min: new_params[i] = b_min
                if new_params[i] > b_max: new_params[i] = b_max
                
                new_score_sub = eval_sub(new_params)
                
                if new_score_sub > current_score_sub:
                    current_params = new_params
                    current_score_sub = new_score_sub
                    improved = True
                    break
            
            if time.time() - start_time > time_limit - 30:
                break
        
        if not improved:
            # Random perturbation
            if time.time() - start_time < time_limit - 60:
                current_params = best_path.ravel().copy() + np.random.normal(0, 0.15, n_params)
                for i in range(n_params):
                    b_min, b_max = bounds[i]
                    current_params[i] = np.clip(current_params[i], b_min, b_max)
                path = current_params.reshape(7, 3)
                for j in range(1, 7):
                    if path[j, 2] < path[j-1, 2]:
                        path[j, 2] = path[j-1, 2]
                current_params = path.ravel()
                current_score_sub = eval_sub(current_params)
                if current_score_sub > best_score_sub:
                    best_score_sub = current_score_sub
                    best_path = path.copy()
            else:
                break
    
    # Final evaluation on full grid
    final_score = _evaluate_path(best_path, grid_x_full, grid_y_full)
    
    # If we have time, try a few more refinements on full grid
    remaining_time = time_limit - (time.time() - start_time)
    if remaining_time > 60:
        # Try small perturbations and evaluate on full grid
        for _ in range(20):
            if time.time() - start_time > time_limit - 10:
                break
            
            perturbed = best_path + np.random.normal(0, 0.05, best_path.shape)
            # Enforce bounds and monotonicity
            for i in range(7):
                perturbed[i, 0] = np.clip(perturbed[i, 0], -2.0, 1.0)
                perturbed[i, 1] = np.clip(perturbed[i, 1], -1.0, 1.0)
                perturbed[i, 2] = np.clip(perturbed[i, 2], 0.0, 90.0)
            for j in range(1, 7):
                if perturbed[j, 2] < perturbed[j-1, 2]:
                    perturbed[j, 2] = perturbed[j-1, 2]
            
            perturbed_score = _evaluate_path(perturbed, grid_x_full, grid_y_full)
            if perturbed_score > final_score:
                best_path = perturbed
                final_score = perturbed_score
    
    return best_path

# EVOLVE-BLOCK-END
