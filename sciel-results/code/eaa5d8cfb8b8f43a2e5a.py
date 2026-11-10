# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def _build_grid():
    xs = np.arange(-3.0, 1.5, 0.01)
    ys = np.arange(-1.5, 1.5, 0.01)
    X, Y = np.meshgrid(xs, ys, indexing='ij')
    return X.ravel(), Y.ravel()

def _build_subgrid(step=3):
    """Build a coarser grid for faster evaluation during optimization."""
    xs = np.arange(-3.0, 1.5, 0.01 * step)
    ys = np.arange(-1.5, 1.5, 0.01 * step)
    X, Y = np.meshgrid(xs, ys, indexing='ij')
    return X.ravel(), Y.ravel()

def _densify_path(path):
    full = np.vstack([[0.0, 0.0, 0.0], path, [-2.0, 0.0, 90.0]])
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

def _evaluate_path(path, grid_x, grid_y):
    dense = _densify_path(path)
    n_points = len(grid_x)
    inside = np.ones(n_points, dtype=bool)
    
    for i in range(len(dense)):
        x, y, a_deg = dense[i]
        a_rad = np.deg2rad(a_deg)
        cos_a = np.cos(a_rad)
        sin_a = np.sin(a_rad)
        
        dx = grid_x - x
        dy = grid_y - y
        u = cos_a * dx + sin_a * dy
        v = -sin_a * dx + cos_a * dy
        
        arm1 = (u >= 0) & (u <= 1) & (v >= -3) & (v <= 1)
        arm2 = (u >= -3) & (u <= 1) & (v >= 0) & (v <= 1)
        in_base = arm1 | arm2
        
        inside = inside & in_base
        if not np.any(inside):
            break
    
    return np.sum(inside) * 0.0001

def _coordinate_descent(params, grid_x, grid_y, deadline, max_rounds=50):
    """Fast coordinate descent optimization using subgrid for speed."""
    n = len(params) // 3
    path = params.copy().reshape(n, 3)
    sub_x, sub_y = _build_subgrid(step=3)
    
    best_area = _evaluate_path(path, sub_x, sub_y)
    
    step_sizes = [0.5, 0.2, 0.1, 0.05, 0.02, 0.01, 0.005]
    
    for step_size in step_sizes:
        if time.time() > deadline:
            break
        improved = True
        while improved and time.time() < deadline:
            improved = False
            for i in range(n):
                for dim in range(3):
                    # Try positive and negative steps
                    for direction in [1, -1]:
                        trial = path.copy()
                        trial[i, dim] += direction * step_size
                        # Keep angles in reasonable range
                        if dim == 2:
                            trial[i, 2] = np.clip(trial[i, 2], -90, 180)
                        area = _evaluate_path(trial, sub_x, sub_y)
                        if area > best_area:
                            path = trial
                            best_area = area
                            improved = True
                            break
    
    # Validate on full grid
    full_x, full_y = _build_grid()
    full_area = _evaluate_path(path, full_x, full_y)
    return path.flatten(), full_area

def moving_sofa_2d() -> np.ndarray:
    start_time = time.time()
    deadline = start_time + 590  # 590s safety margin
    
    # Use subgrid for fast optimization
    sub_x, sub_y = _build_subgrid(step=3)
    
    # Initial path
    n_ctrl = 8
    t = np.linspace(0, 1, n_ctrl)
    angles = np.linspace(10, 80, n_ctrl)
    x = -2.0 * t
    y = 0.5 * np.sin(np.pi * t)
    
    initial_params = np.column_stack([x, y, angles]).flatten()
    
    best_params = initial_params.copy()
    best_area = _evaluate_path(initial_params.reshape(n_ctrl, 3), sub_x, sub_y)
    
    # Strategy 1: Nelder-Mead on subgrid
    try:
        result = minimize(
            lambda p: -_evaluate_path(p.reshape(n_ctrl, 3), sub_x, sub_y),
            initial_params,
            method='Nelder-Mead',
            options={'maxiter': 150, 'xatol': 0.05, 'fatol': 0.001}
        )
        if result.fun < 0:
            area = -result.fun
            if area > best_area:
                best_area = area
                best_params = result.x.copy()
    except:
        pass
    
    if time.time() > deadline:
        # Validate on full grid before returning
        full_x, full_y = _build_grid()
        final_area = _evaluate_path(best_params.reshape(n_ctrl, 3), full_x, full_y)
        return best_params.reshape(n_ctrl, 3)
    
    # Strategy 2: Coordinate descent on subgrid
    params_cd, area_cd = _coordinate_descent(best_params, sub_x, sub_y, deadline)
    if area_cd > best_area:
        best_area = area_cd
        best_params = params_cd
    
    if time.time() > deadline:
        full_x, full_y = _build_grid()
        return best_params.reshape(n_ctrl, 3)
    
    # Strategy 3: Try with more control points on subgrid
    n_ctrl2 = 12
    t2 = np.linspace(0, 1, n_ctrl2)
    angles2 = np.linspace(8, 82, n_ctrl2)
    x2 = -2.0 * t2
    y2 = 0.5 * np.sin(np.pi * t2)
    initial_params2 = np.column_stack([x2, y2, angles2]).flatten()
    
    try:
        result = minimize(
            lambda p: -_evaluate_path(p.reshape(n_ctrl2, 3), sub_x, sub_y),
            initial_params2,
            method='Nelder-Mead',
            options={'maxiter': 150, 'xatol': 0.05, 'fatol': 0.001}
        )
        if result.fun < 0:
            area = -result.fun
            # Compare on subgrid first
            if area > best_area:
                # Validate on full grid
                full_x, full_y = _build_grid()
                full_area = _evaluate_path(result.x.reshape(n_ctrl2, 3), full_x, full_y)
                if full_area > best_area:
                    best_area = full_area
                    best_params = result.x.copy()
                    n_ctrl = n_ctrl2
    except:
        pass
    
    if time.time() > deadline:
        full_x, full_y = _build_grid()
        return best_params.reshape(n_ctrl, 3)
    
    # Final refinement with coordinate descent on subgrid
    params_final, area_final = _coordinate_descent(best_params, sub_x, sub_y, deadline)
    if area_final > best_area:
        best_area = area_final
        best_params = params_final
    
    # Final validation on full grid
    full_x, full_y = _build_grid()
    final_area = _evaluate_path(best_params.reshape(n_ctrl, 3), full_x, full_y)
    
    return best_params.reshape(n_ctrl, 3)

# EVOLVE-BLOCK-END
