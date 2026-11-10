import numpy as np
import time
import scipy.optimize as so

def moving_sofa_2d() -> np.ndarray:
    # Grid definition
    x_grid = np.arange(-3.0, 1.5, 0.01)
    y_grid = np.arange(-1.5, 1.5, 0.01)
    
    X, Y = np.meshgrid(x_grid, y_grid, indexing='ij')
    X_flat = X.flatten()
    Y_flat = Y.flatten()
    n_points = len(X_flat)
    
    # Subsample for fast optimization (every 4th point)
    step = 4
    x_sub = x_grid[::step]
    y_sub = y_grid[::step]
    X_sub, Y_sub = np.meshgrid(x_sub, y_sub, indexing='ij')
    X_sub_flat = X_sub.flatten()
    Y_sub_flat = Y_sub.flatten()
    
    # Precompute for fast evaluation
    X_col_sub = X_sub_flat[:, np.newaxis]
    Y_col_sub = Y_sub_flat[:, np.newaxis]
    
    # Precompute for full evaluation
    X_col_full = X_flat[:, np.newaxis]
    Y_col_full = Y_flat[:, np.newaxis]

    def evaluate_path_fast(path):
        # path is (n_ctrl, 3)
        full_path = np.vstack([
            [[0.0, 0.0, 0.0]],
            path,
            [[-2.0, 0.0, 90.0]]
        ])
        
        # Generate all placements
        placements = []
        for i in range(len(full_path) - 1):
            p1 = full_path[i]
            p2 = full_path[i+1]
            
            da = p2[2] - p1[2]
            m = max(1, int(np.ceil(abs(da))))
            
            ts = np.linspace(0, 1, m+1)
            for t in ts:
                x = p1[0] + t * (p2[0] - p1[0])
                y = p1[1] + t * (p2[1] - p1[1])
                a = p1[2] + t * (p2[2] - p1[2])
                placements.append((x, y, a))
        
        if not placements:
            return 0.0
            
        px = np.array([p[0] for p in placements])[np.newaxis, :]
        py = np.array([p[1] for p in placements])[np.newaxis, :]
        pa = np.array([p[2] for p in placements])[np.newaxis, :]
        
        theta = -np.deg2rad(pa)
        c = np.cos(theta)
        s = np.sin(theta)
        
        # Use subsampled grid for speed
        wx = X_col_sub - px
        wy = Y_col_sub - py
        
        u = wx * c - wy * s
        v = wx * s + wy * c
        
        in_r1 = (u >= 0.0) & (u <= 1.0) & (v >= -3.0) & (v <= 1.0)
        in_r2 = (u >= -3.0) & (u <= 1.0) & (v >= 0.0) & (v <= 1.0)
        
        inside_any = in_r1 | in_r2
        
        # Accumulate validity
        valid_points = np.all(inside_any, axis=1)
        
        return np.sum(valid_points) * (step * step * 0.0001) # Scale back to area

    def evaluate_path_full(path):
        full_path = np.vstack([
            [[0.0, 0.0, 0.0]],
            path,
            [[-2.0, 0.0, 90.0]]
        ])
        
        placements = []
        for i in range(len(full_path) - 1):
            p1 = full_path[i]
            p2 = full_path[i+1]
            
            da = p2[2] - p1[2]
            m = max(1, int(np.ceil(abs(da))))
            
            ts = np.linspace(0, 1, m+1)
            for t in ts:
                x = p1[0] + t * (p2[0] - p1[0])
                y = p1[1] + t * (p2[1] - p1[1])
                a = p1[2] + t * (p2[2] - p1[2])
                placements.append((x, y, a))
        
        if not placements:
            return 0.0
            
        px = np.array([p[0] for p in placements])[np.newaxis, :]
        py = np.array([p[1] for p in placements])[np.newaxis, :]
        pa = np.array([p[2] for p in placements])[np.newaxis, :]
        
        theta = -np.deg2rad(pa)
        c = np.cos(theta)
        s = np.sin(theta)
        
        wx = X_col_full - px
        wy = Y_col_full - py
        
        u = wx * c - wy * s
        v = wx * s + wy * c
        
        in_r1 = (u >= 0.0) & (u <= 1.0) & (v >= -3.0) & (v <= 1.0)
        in_r2 = (u >= -3.0) & (u <= 1.0) & (v >= 0.0) & (v <= 1.0)
        
        inside_any = in_r1 | in_r2
        
        valid_points = np.all(inside_any, axis=1)
        
        return np.sum(valid_points) * 0.0001

    # Optimization
    n_ctrl = 16
    ctrl_angles = np.linspace(5.0, 85.0, n_ctrl)
    
    def make_path_from_params(params):
        path = np.zeros((n_ctrl, 3))
        path[:, 2] = ctrl_angles
        path[:, 0] = params[0::2]
        path[:, 1] = params[1::2]
        return path

    # Initial guess: Linear interpolation of x, y with a slight bulge
    current_params = np.zeros(2 * n_ctrl)
    for i in range(n_ctrl):
        t = ctrl_angles[i] / 90.0
        # Move from (0,0) to (-2,0)
        current_params[2*i] = -2.0 * t
        current_params[2*i+1] = 0.0
        
    bounds = []
    for i in range(n_ctrl):
        bounds.append((-3.0, 1.0))
        bounds.append((-2.0, 2.0))
        
    def objective_fast(params):
        path = make_path_from_params(params)
        return -evaluate_path_fast(path)
        
    def objective_full(params):
        path = make_path_from_params(params)
        return -evaluate_path_full(path)

    start_time = time.time()
    time_limit = 600.0
    deadline = start_time + time_limit

    best_params = current_params
    best_score = -objective_fast(current_params)

    # Phase 1: Differential Evolution on Fast Objective
    try:
        # Use a callback to check time
        def de_callback(xk, convergence):
            nonlocal deadline
            return time.time() > deadline - 60.0 # Stop 60s early for polish

        result = so.differential_evolution(
            objective_fast, 
            bounds, 
            maxiter=100, 
            popsize=20, 
            tol=1e-5, 
            seed=42,
            callback=de_callback
        )
        
        if result.fun < objective_fast(best_params):
            best_params = result.x
            best_score = -result.fun

    except Exception as e:
        pass

    # Phase 2: Polish with Nelder-Mead on Full Objective
    if time.time() < deadline - 30:
        def nm_objective(params):
            return objective_full(params)
            
        # Limit iterations to ensure we don't time out
        max_iter_nm = 200
        try:
            result_nm = so.minimize(
                nm_objective, 
                best_params, 
                method='Nelder-Mead', 
                options={'maxiter': max_iter_nm, 'xatol': 1e-5, 'fatol': 1e-7}
            )
            
            # Check if the result is better
            score_nm = -result_nm.fun
            if score_nm > best_score:
                best_params = result_nm.x
                best_score = score_nm
        except Exception:
            pass

    # Final Check
    final_path = make_path_from_params(best_params)
    if not np.all(np.isfinite(final_path)):
        final_path = make_path_from_params(current_params)
        
    return final_path
