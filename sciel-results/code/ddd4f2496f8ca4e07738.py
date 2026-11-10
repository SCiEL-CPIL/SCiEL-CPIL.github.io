import numpy as np
import time
import scipy.optimize as so

def moving_sofa_2d() -> np.ndarray:
    """
    Optimizes the path of the moving sofa to maximize the area of the intersection.
    """
    start_time = time.time()
    time_limit = 600.0
    safety_margin = 40.0

    # Sofa points definition - full grid
    i_vals = np.arange(-3.0, 1.5, 0.01)
    j_vals = np.arange(-1.5, 1.5, 0.01)
    P_x_full, P_y_full = np.meshgrid(i_vals, j_vals, indexing='xy')
    P_x_full = P_x_full.flatten()
    P_y_full = P_y_full.flatten()
    N_pts_full = len(P_x_full)

    # Coarser subgrid for fast evaluation during optimization
    # Use every 3rd point in each dimension (~1/9 the points)
    coarse_idx = slice(0, N_pts_full, 3)
    P_x_coarse = P_x_full[coarse_idx]
    P_y_coarse = P_y_full[coarse_idx]
    N_pts_coarse = len(P_x_coarse)

    # Pre-compute the dense path structure for a given set of angles
    def build_dense_structure(angles):
        """Build the list of (x, y, a) placements for a given path."""
        full_path = [(0.0, 0.0, 0.0)] + list(angles) + [(-2.0, 0.0, 90.0)]
        
        densified = [full_path[0]]
        for i in range(len(full_path) - 1):
            x1, y1, a1 = full_path[i]
            x2, y2, a2 = full_path[i+1]
            
            da = abs(a2 - a1)
            m = max(1, int(np.ceil(da)))
            
            if m > 1:
                for k in range(1, m):
                    t = k / m
                    x = x1 + t * (x2 - x1)
                    y = y1 + t * (y2 - y1)
                    a = a1 + t * (a2 - a1)
                    densified.append((x, y, a))
        
        densified.append(full_path[-1])
        return densified

    # Fast evaluation using coarse grid
    def evaluate_path_coarse(path):
        """Evaluate area using coarse subgrid for speed."""
        densified = build_dense_structure(path)
        
        in_sofa = np.ones(N_pts_coarse, dtype=bool)
        
        for x, y, a in densified:
            theta = np.deg2rad(a)
            cos_t = np.cos(theta)
            sin_t = np.sin(theta)
            
            dx = P_x_coarse - x
            dy = P_y_coarse - y
            
            u = dx * cos_t + dy * sin_t
            v = -dx * sin_t + dy * cos_t
            
            cond1 = (u >= 0) & (u <= 1) & (v >= -3) & (v <= 1)
            cond2 = (u >= -3) & (u <= 1) & (v >= 0) & (v <= 1)
            mask = cond1 | cond2
            
            in_sofa &= mask
            
            if not np.any(in_sofa):
                return 0.0
        
        return np.sum(in_sofa) * 0.0001

    # Full evaluation for final verification
    def evaluate_path_full(path):
        """Evaluate area using full grid."""
        densified = build_dense_structure(path)
        
        in_sofa = np.ones(N_pts_full, dtype=bool)
        
        for x, y, a in densified:
            theta = np.deg2rad(a)
            cos_t = np.cos(theta)
            sin_t = np.sin(theta)
            
            dx = P_x_full - x
            dy = P_y_full - y
            
            u = dx * cos_t + dy * sin_t
            v = -dx * sin_t + dy * cos_t
            
            cond1 = (u >= 0) & (u <= 1) & (v >= -3) & (v <= 1)
            cond2 = (u >= -3) & (u <= 1) & (v >= 0) & (v <= 1)
            mask = cond1 | cond2
            
            in_sofa &= mask
            
            if not np.any(in_sofa):
                return 0.0
        
        return np.sum(in_sofa) * 0.0001

    # Parameterization
    angles = np.array([10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 85.0, 88.0])
    n_angles = len(angles)
    n_params = 2 * n_angles
    
    # Initial guess from baseline
    base_x = [-0.1, -0.2, -0.3, -0.3, -0.4, -0.5, -0.6, -0.7]
    base_y = [0.1, 0.2, 0.3, 0.3, 0.3, 0.2, 0.2, 0.3]
    base_a = [15.0, 20.0, 34.0, 40.0, 50.0, 55.0, 75.0, 86.0]
    
    init_x = np.interp(angles, base_a, base_x)
    init_y = np.interp(angles, base_a, base_y)
    
    p0 = np.concatenate([init_x, init_y])

    def time_guard():
        return time.time() - start_time > (time_limit - safety_margin)

    # Objective function for coarse evaluation (minimize negative area)
    def objective_coarse(params):
        x_vals = params[:n_angles]
        y_vals = params[n_angles:]
        path = list(zip(x_vals, y_vals, angles))
        area = evaluate_path_coarse(path)
        return -area

    # Objective function for full evaluation
    def objective_full(params):
        x_vals = params[:n_angles]
        y_vals = params[n_angles:]
        path = list(zip(x_vals, y_vals, angles))
        area = evaluate_path_full(path)
        return -area

    # Optimize using coarse grid first
    best_area = objective_coarse(p0)
    best_params = p0.copy()

    # Phase 1: Powell on coarse grid
    if not time_guard():
        try:
            res = so.minimize(objective_coarse, p0, method='Powell',
                              options={'maxiter': 3000, 'maxfev': 3000, 'disp': False})
            if not time_guard():
                if res.fun < best_area:
                    best_area = -res.fun
                    best_params = res.x.copy()
        except Exception:
            pass

    # Phase 2: Nelder-Mead on coarse grid
    if not time_guard():
        try:
            res2 = so.minimize(objective_coarse, best_params, method='Nelder-Mead',
                               options={'maxiter': 2000, 'maxfev': 2000, 'disp': False})
            if not time_guard():
                if res2.fun < best_area:
                    best_area = -res2.fun
                    best_params = res2.x.copy()
        except Exception:
            pass

    # Phase 3: Coordinate descent with multiple step sizes on coarse grid
    if not time_guard():
        current_params = best_params.copy()
        current_area = -best_area
        step_sizes = [0.1, 0.05, 0.02, 0.01, 0.005, 0.002, 0.001]
        
        for step in step_sizes:
            if time_guard():
                break
            improved = True
            while improved and not time_guard():
                improved = False
                for i in range(n_params):
                    for delta in [step, -step]:
                        trial = current_params.copy()
                        trial[i] += delta
                        trial_area = -objective_coarse(trial)
                        if trial_area > current_area:
                            current_params = trial
                            current_area = trial_area
                            improved = True
                            if -current_area < best_area:
                                best_area = -current_area
                                best_params = current_params.copy()

    # Phase 4: Fine-tune with full grid evaluation
    if not time_guard():
        full_best_area = objective_full(best_params)
        full_best_params = best_params.copy()
        
        # Coordinate descent on full grid with small steps
        step_sizes_full = [0.01, 0.005, 0.002, 0.001]
        
        for step in step_sizes_full:
            if time_guard():
                break
            improved = True
            while improved and not time_guard():
                improved = False
                for i in range(n_params):
                    for delta in [step, -step]:
                        trial = full_best_params.copy()
                        trial[i] += delta
                        trial_area = -objective_full(trial)
                        if trial_area > full_best_area:
                            full_best_params = trial
                            full_best_area = trial_area
                            improved = True
                            if -full_best_area < best_area:
                                best_area = -full_best_area
                                best_params = full_best_params.copy()

    # Construct final path
    x_vals = best_params[:n_angles]
    y_vals = best_params[n_angles:]
    
    final_path = []
    for i in range(n_angles):
        final_path.append([x_vals[i], y_vals[i], angles[i]])
        
    path_arr = np.array(final_path, dtype=np.float64)
    
    return path_arr
