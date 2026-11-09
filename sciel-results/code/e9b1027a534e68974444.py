import numpy as np
import time
from scipy.optimize import minimize

def moving_sofa_2d() -> np.ndarray:
    t_start = time.time()
    time_limit = 600.0
    margin = 30.0  # Safety margin
    
    # Pre-compute the sofa grid
    i = np.arange(450)
    j = np.arange(300)
    X, Y = np.meshgrid(i, j, indexing='ij')
    x_grid = -3 + 0.01 * X
    y_grid = -1.5 + 0.01 * Y
    x_flat = x_grid.ravel()
    y_flat = y_grid.ravel()
    n_points = x_flat.size

    # Pre-compute rotation parameters for 91 steps (0 to 90 degrees)
    a_vals = np.arange(91)
    rad_vals = np.deg2rad(a_vals)
    cos_vals = np.cos(rad_vals)
    sin_vals = np.sin(rad_vals)
    
    # t values for polynomial evaluation: 0 to 1 in 91 steps
    t_vals = np.linspace(0, 1, 91)
    t_powers = t_vals[:, None] ** np.arange(6)  # shape (91, 6)

    def compute_score(params):
        c2, c3, c4, c5, d2, d3, d4, d5 = params
        
        c1 = -2.0 - (c2 + c3 + c4 + c5)
        d1 = -(d2 + d3 + d4 + d5)
        
        coeffs_x = np.array([c1, c2, c3, c4, c5])
        coeffs_y = np.array([d1, d2, d3, d4, d5])
        
        # Evaluate polynomials at all 91 t values
        # t_powers[:, 1:] has shape (91, 5)
        xs = t_powers[:, 1:] @ coeffs_x
        ys = t_powers[:, 1:] @ coeffs_y
        
        # Vectorized check for all angles and all points
        # dx, dy shape: (91, 135000)
        dx = x_flat[None, :] - xs[:, None]
        dy = y_flat[None, :] - ys[:, None]
        
        # Rotation
        u = dx * cos_vals[:, None] + dy * sin_vals[:, None]
        v = -dx * sin_vals[:, None] + dy * cos_vals[:, None]
        
        cond1 = (u >= -3) & (u <= 1) & (v >= 0) & (v <= 1)
        cond2 = (u >= 0) & (u <= 1) & (v >= -3) & (v <= 1)
        
        valid_all = (cond1 | cond2).all(axis=0)
        
        count = np.sum(valid_all)
        return count * 0.0001

    def objective(params):
        return -compute_score(params)

    best_score = 0.0
    best_params = np.zeros(8)
    
    # Strategy: Use a few random restarts with local optimization (Nelder-Mead)
    # This is often faster and more reliable for 8D problems than DE with strict time limits.
    # We will run as many restarts as possible within the time budget.
    
    rng = np.random.default_rng(42)
    
    # Initial guess
    current_params = rng.uniform(-2, 2, size=8)
    current_score = -objective(current_params)
    
    if current_score > best_score:
        best_score = current_score
        best_params = current_params.copy()
        
    while time.time() - t_start < time_limit - margin:
        # Random restart
        start_params = rng.uniform(-3, 3, size=8)
        
        # Local optimization
        try:
            res = minimize(
                objective, 
                start_params, 
                method='Nelder-Mead', 
                options={'maxiter': 500, 'xatol': 1e-5, 'fatol': 1e-6}
            )
            score = -res.fun
            if score > best_score:
                best_score = score
                best_params = res.x
        except Exception:
            pass
            
        # Check time again
        if time.time() - t_start > time_limit - margin:
            break

    # Construct the final path
    c2, c3, c4, c5, d2, d3, d4, d5 = best_params
    c1 = -2.0 - (c2 + c3 + c4 + c5)
    d1 = -(d2 + d3 + d4 + d5)
    
    coeffs_x = np.array([c1, c2, c3, c4, c5])
    coeffs_y = np.array([d1, d2, d3, d4, d5])
    
    xs = t_powers[:, 1:] @ coeffs_x
    ys = t_powers[:, 1:] @ coeffs_y
    
    path_pts = np.column_stack((xs, ys, a_vals))
    
    # Ensure finite
    if not np.all(np.isfinite(path_pts)):
        # Fallback to a simple valid path if something went wrong
        # x: 0 -> -2, y: 0 -> 0, a: 0 -> 90
        xs_fallback = np.linspace(0, -2, 91)
        ys_fallback = np.zeros(91)
        path_pts = np.column_stack((xs_fallback, ys_fallback, a_vals))

    return path_pts
