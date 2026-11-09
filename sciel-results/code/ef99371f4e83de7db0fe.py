import numpy as np
import time
from scipy.optimize import differential_evolution, minimize

def moving_sofa_2d() -> np.ndarray:
    t_start = time.time()
    time_limit = 600.0
    margin = 30.0  # Safety margin

    # Pre-compute the sofa grid
    # i = 0..449, j = 0..299
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
    t_vals = a_vals / 90.0
    rad_vals = np.deg2rad(a_vals)
    cos_vals = np.cos(rad_vals)
    sin_vals = np.sin(rad_vals)

    # Power terms for t
    t_powers = t_vals[:, None] ** np.arange(6)  # shape (91, 6), powers 0..5

    def compute_score(params):
        # params: [c2, c3, c4, c5, d2, d3, d4, d5]
        # x(t) = c1*t + c2*t^2 + c3*t^3 + c4*t^4 + c5*t^5
        # y(t) = d1*t + d2*t^2 + d3*t^3 + d4*t^4 + d5*t^5
        
        c2, c3, c4, c5, d2, d3, d4, d5 = params
        
        # c1 = -2 - (c2+c3+c4+c5)
        # d1 = -(d2+d3+d4+d5)
        
        # Coefficients for x: [0, c1, c2, c3, c4, c5]
        # We want x(t) = c1*t + c2*t^2 + ...
        # Let's construct the matrix multiplication
        # t_powers shape (91, 6). Columns are 1, t, t^2, t^3, t^4, t^5
        # We want to ignore the 1st column (power 0) and use 1..5
        # coeffs_x = [c1, c2, c3, c4, c5]
        # coeffs_y = [d1, d2, d3, d4, d5]
        
        c1 = -2.0 - (c2 + c3 + c4 + c5)
        d1 = -(d2 + d3 + d4 + d5)
        
        coeffs_x = np.array([c1, c2, c3, c4, c5])
        coeffs_y = np.array([d1, d2, d3, d4, d5])
        
        # t_powers[:, 1:] has shape (91, 5)
        # xs = t_powers[:, 1:] @ coeffs_x  -> shape (91,)
        xs = t_powers[:, 1:] @ coeffs_x
        ys = t_powers[:, 1:] @ coeffs_y
        
        mask = np.ones(n_points, dtype=bool)
        
        # Process in chunks to manage memory
        chunk_size = 10
        for start_k in range(0, 91, chunk_size):
            end_k = min(start_k + chunk_size, 91)
            k_slice = slice(start_k, end_k)
            
            xs_chunk = xs[k_slice]
            ys_chunk = ys[k_slice]
            cos_chunk = cos_vals[k_slice]
            sin_chunk = sin_vals[k_slice]
            
            dx = x_flat[None, :] - xs_chunk[:, None]
            dy = y_flat[None, :] - ys_chunk[:, None]
            
            u = dx * cos_chunk[:, None] + dy * sin_chunk[:, None]
            v = -dx * sin_chunk[:, None] + dy * cos_chunk[:, None]
            
            cond1 = (u >= -3) & (u <= 1) & (v >= 0) & (v <= 1)
            cond2 = (u >= 0) & (u <= 1) & (v >= -3) & (v <= 1)
            valid_chunk = cond1 | cond2
            
            valid_all = valid_chunk.all(axis=0)
            mask &= valid_all
            
            if not np.any(mask):
                return 0.0
        
        count = np.sum(mask)
        return count * 0.0001

    def objective(params):
        return -compute_score(params)

    # Define bounds for the parameters
    # The path should stay roughly within the corridor bounds to be effective.
    # x goes from 0 to -2. y goes from 0 to 0.
    # Intermediate values shouldn't be too wild.
    bounds = [(-5, 5)] * 8
    
    # Initial best
    best_score = 0.0
    best_params = np.zeros(8)
    
    # 1. Global Optimization: Differential Evolution
    # Use a smaller population to save time, but enough to explore
    # maxiter is the number of generations
    # tol is the relative tolerance
    if time.time() - t_start < time_limit - margin:
        try:
            res = differential_evolution(
                objective, 
                bounds, 
                maxiter=50, 
                popsize=10, 
                tol=1e-4, 
                seed=42,
                polish=False  # We will do our own polish
            )
            score = -res.fun
            if score > best_score:
                best_score = score
                best_params = res.x
        except Exception:
            pass

    # 2. Local Refinement: Nelder-Mead
    if time.time() - t_start < time_limit - margin:
        try:
            res2 = minimize(
                objective, 
                best_params, 
                method='Nelder-Mead', 
                options={'maxiter': 1000, 'xatol': 1e-6, 'fatol': 1e-8}
            )
            score = -res2.fun
            if score > best_score:
                best_score = score
                best_params = res2.x
        except Exception:
            pass

    # 3. Try a few random restarts if time permits?
    # DE is usually good enough. Let's stick to one good global + local.
    
    # Construct the final path
    c2, c3, c4, c5, d2, d3, d4, d5 = best_params
    c1 = -2.0 - (c2 + c3 + c4 + c5)
    d1 = -(d2 + d3 + d4 + d5)
    
    coeffs_x = np.array([c1, c2, c3, c4, c5])
    coeffs_y = np.array([d1, d2, d3, d4, d5])
    
    xs = t_powers[:, 1:] @ coeffs_x
    ys = t_powers[:, 1:] @ coeffs_y
    
    path_pts = np.column_stack((xs, ys, a_vals))
    
    return path_pts
