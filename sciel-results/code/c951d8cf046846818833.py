# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def _base_corridor_contains(u, v):
    """Check if point (u, v) is in the base L-shaped corridor."""
    # Union of {0 <= u <= 1, -3 <= v <= 1} and {-3 <= u <= 1, 0 <= v <= 1}
    in_h = (0 <= u) & (u <= 1) & (-3 <= v) & (v <= 1)
    in_v = (-3 <= u) & (u <= 1) & (0 <= v) & (v <= 1)
    return in_h | in_v

def _evaluate_path(path):
    """Evaluate the area of the sofa for a given path."""
    # Full path with start and end
    start = np.array([0.0, 0.0, 0.0])
    end = np.array([-2.0, 0.0, 90.0])
    
    # Build densified path
    all_points = [start]
    for i in range(len(path)):
        all_points.append(path[i])
    all_points.append(end)
    
    # Densify: between consecutive rows, add linear interpolations
    densified = [all_points[0]]
    for i in range(len(all_points) - 1):
        p1 = all_points[i]
        p2 = all_points[i + 1]
        da = abs(p2[2] - p1[2])
        m = max(1, int(np.ceil(da)))
        for k in range(1, m + 1):
            t = k / m
            interp = p1 + t * (p2 - p1)
            densified.append(interp)
    
    # Grid of sofa points
    u_vals = np.arange(-3, 1.5, 0.01)
    v_vals = np.arange(-1.5, 1.5, 0.01)
    U, V = np.meshgrid(u_vals, v_vals, indexing='ij')
    grid_points = np.column_stack([U.ravel(), V.ravel()])
    
    # Check each grid point against all placements
    n_points = len(grid_points)
    inside_all = np.ones(n_points, dtype=bool)
    
    for (x, y, a_deg) in densified:
        a_rad = np.radians(a_deg)
        cos_a = np.cos(a_rad)
        sin_a = np.sin(a_rad)
        
        # R(-a) rotates by -a
        # R(-a)(w - (x,y)) = [cos(a)*(wx-x) + sin(a)*(wy-y), -sin(a)*(wx-x) + cos(a)*(wy-y)]
        dx = grid_points[:, 0] - x
        dy = grid_points[:, 1] - y
        
        ru = cos_a * dx + sin_a * dy
        rv = -sin_a * dx + cos_a * dy
        
        in_this = _base_corridor_contains(ru, rv)
        inside_all &= in_this
        
        if not inside_all.any():
            break
    
    return inside_all.sum() * 0.0001

def _objective(params):
    """Negative area for minimization."""
    # params: x0, y0, x1, y1, x2, y2, x3, y3, x4, y4, x5, y5, x6, y6, x7, y7
    # Angles are fixed to a reasonable progression
    x_coords = params[:8]
    y_coords = params[8:16]
    
    # Fixed angle progression from ~15 to ~86 degrees
    angles = np.array([15.0, 20.0, 34.0, 40.0, 50.0, 55.0, 75.0, 86.0])
    
    path = np.column_stack([x_coords, y_coords, angles])
    
    try:
        area = _evaluate_path(path)
    except Exception:
        return 1e10
    
    return -area

def moving_sofa_2d() -> np.ndarray:
    """
    Find an optimal path for the moving sofa problem.
    
    Returns:
        path: np.ndarray of shape (n, 3); each row is (x, y, angle_degrees)
    """
    deadline = time.time() + 600  # 600 second deadline
    
    # Start with the baseline path
    best_path = np.array([
        [-0.1, 0.1, 15.0],
        [-0.2, 0.2, 20.0],
        [-0.3, 0.3, 34.0],
        [-0.3, 0.3, 40.0],
        [-0.4, 0.3, 50.0],
        [-0.5, 0.2, 55.0],
        [-0.6, 0.2, 75.0],
        [-0.7, 0.3, 86.0],
    ], dtype=np.float64)
    
    best_area = _evaluate_path(best_path)
    
    # Try to optimize the x,y coordinates while keeping angles fixed
    x0 = best_path[:, 0]
    y0 = best_path[:, 1]
    initial_params = np.concatenate([x0, y0])
    
    # Use Nelder-Mead optimization
    try:
        result = minimize(
            _objective,
            initial_params,
            method='Nelder-Mead',
            options={'maxiter': 200, 'xatol': 1e-4, 'fatol': 1e-6}
        )
        
        if time.time() < deadline - 60:
            if result.success and result.fun < -best_area:
                opt_params = result.x
                opt_x = opt_params[:8]
                opt_y = opt_params[8:16]
                angles = np.array([15.0, 20.0, 34.0, 40.0, 50.0, 55.0, 75.0, 86.0])
                new_path = np.column_stack([opt_x, opt_y, angles])
                new_area = _evaluate_path(new_path)
                if new_area > best_area:
                    best_path = new_path
                    best_area = new_area
    except Exception:
        pass
    
    # Try a few manual perturbations if time allows
    if time.time() < deadline - 120:
        angles = np.array([15.0, 20.0, 34.0, 40.0, 50.0, 55.0, 75.0, 86.0])
        
        # Try shifting the whole path
        for dx, dy in [(-0.05, 0.05), (0.05, -0.05), (-0.1, 0), (0.1, 0), (0, -0.1), (0, 0.1)]:
            test_path = best_path.copy()
            test_path[:, 0] += dx
            test_path[:, 1] += dy
            test_area = _evaluate_path(test_path)
            if test_area > best_area:
                best_path = test_path
                best_area = test_area
    
    return best_path

# EVOLVE-BLOCK-END
