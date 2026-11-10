import numpy as np
import time

def moving_sofa_2d() -> np.ndarray:
    # Grid definition
    x_grid = np.arange(-3.0, 1.5, 0.01)
    y_grid = np.arange(-1.5, 1.5, 0.01)
    X, Y = np.meshgrid(x_grid, y_grid, indexing='ij')
    X_flat = X.flatten()
    Y_flat = Y.flatten()
    n_pts = len(X_flat)
    
    # Precompute for speed
    def check_inside_fast(x, y, a_deg):
        theta = -np.deg2rad(a_deg)
        c = np.cos(theta)
        s = np.sin(theta)
        wx = X_flat - x
        wy = Y_flat - y
        u = wx * c - wy * s
        v = wx * s + wy * c
        in_r1 = (u >= 0.0) & (u <= 1.0) & (v >= -3.0) & (v <= 1.0)
        in_r2 = (u >= -3.0) & (u <= 1.0) & (v >= 0.0) & (v <= 1.0)
        return in_r1 | in_r2

    def evaluate_path_coarse(path):
        """Evaluate with reduced grid for speed"""
        # Use every 3rd point for coarse evaluation
        stride = 3
        Xc = X_flat[::stride]
        Yc = Y_flat[::stride]
        
        full_path = np.vstack([
            [[0.0, 0.0, 0.0]],
            path,
            [[-2.0, 0.0, 90.0]]
        ])
        
        mask = np.ones(len(Xc), dtype=bool)
        
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
                
                theta = -np.deg2rad(a)
                c = np.cos(theta)
                s = np.sin(theta)
                wx = Xc - x
                wy = Yc - y
                u = wx * c - wy * s
                v = wx * s + wy * c
                in_r1 = (u >= 0.0) & (u <= 1.0) & (v >= -3.0) & (v <= 1.0)
                in_r2 = (u >= -3.0) & (u <= 1.0) & (v >= 0.0) & (v <= 1.0)
                inside = in_r1 | in_r2
                mask &= inside
                if not mask.any():
                    return 0.0
        
        return np.sum(mask) * 0.0001 * stride

    def evaluate_path_full(path):
        """Full evaluation"""
        full_path = np.vstack([
            [[0.0, 0.0, 0.0]],
            path,
            [[-2.0, 0.0, 90.0]]
        ])
        
        mask = np.ones(n_pts, dtype=bool)
        
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
                
                inside = check_inside_fast(x, y, a)
                mask &= inside
                if not mask.any():
                    return 0.0
        
        return np.sum(mask) * 0.0001

    # Parameterization: N waypoints with x, y, angle
    # Angles are constrained to be monotonically increasing from 0 to 90
    N_WAYPOINTS = 12
    
    def params_to_path(params):
        """Convert params to path. Params: x[0:N], y[0:N], angle_offsets[0:N]
        Angles are spread evenly with small offsets"""
        n = len(params) // 3
        xs = params[0:n]
        ys = params[n:2*n]
        angle_offs = params[2*n:3*n]
        
        # Base angles: evenly spaced from 5 to 85
        base_angles = np.linspace(5.0, 85.0, n)
        # Add small offsets (clamped to keep monotonic)
        angles = base_angles + angle_offs * 0.5
        # Ensure monotonic and within (0, 90)
        for i in range(1, n):
            if angles[i] <= angles[i-1]:
                angles[i] = angles[i-1] + 1.0
        angles = np.clip(angles, 1.0, 89.0)
        
        path = np.column_stack([xs, ys, angles])
        return path

    # Initial guess based on geometric reasoning
    # The sofa turns around the inner corner
    # Start near (0,0), end near (-2,0)
    # The path should arc around the corner
    n = N_WAYPOINTS
    t = np.linspace(0, 1, n)
    # Arc path: center around (-1, 1) with radius ~1.5
    cx, cy = -1.0, 1.0
    r = 1.5
    # Start angle: pointing right, end angle: pointing down-left
    # Parametrize: angle from 0 to 90 degrees (in path space)
    theta = np.deg2rad(t * 90.0)
    xs_init = cx + r * np.cos(theta) - 0.5
    ys_init = cy - r * np.sin(theta) + 0.3
    
    # Refine initial positions
    xs_init = xs_init - 0.3 * t  # drift left
    ys_init = ys_init + 0.2 * t  # slight up
    
    angle_offs_init = np.zeros(n)
    
    init_params = np.concatenate([xs_init, ys_init, angle_offs_init])
    
    start_time = time.time()
    time_limit = 600.0
    
    def objective(params):
        if time.time() - start_time > time_limit:
            return 0.0
        path = params_to_path(params)
        return -evaluate_path_coarse(path)

    # Nelder-Mead implementation
    def nelder_mead(func, x0, args=(), xtol=1e-4, ftol=1e-5, maxfev=3000, 
                    alpha=1.0, gamma=2.0, rho=0.5, sigma=0.5):
        n = len(x0)
        simplex = np.zeros((n+1, n))
        simplex[0] = x0
        for i in range(n):
            simplex[i+1] = x0.copy()
            simplex[i+1, i] = x0[i] + 0.1 * abs(x0[i]) if x0[i] != 0 else 0.1
        
        fvals = np.array([func(x) for x in simplex])
        fev = n + 1
        
        while fev < maxfev:
            if time.time() - start_time > time_limit:
                break
            
            idx = np.argsort(fvals)
            simplex = simplex[idx]
            fvals = fvals[idx]
            
            if np.max(np.abs(fvals - fvals[0])) < ftol:
                break
            if np.max(np.abs(simplex[0] - simplex[1])) < xtol:
                break
            
            centroid = np.mean(simplex[:-1], axis=0)
            xr = centroid + alpha * (centroid - simplex[-1])
            fr = func(xr)
            fev += 1
            
            if fr < fvals[0]:
                xe = centroid + gamma * (xr - centroid)
                fe = func(xe)
                fev += 1
                if fe < fr:
                    simplex[-1] = xe
                    fvals[-1] = fe
                else:
                    simplex[-1] = xr
                    fvals[-1] = fr
            elif fr < fvals[-2]:
                simplex[-1] = xr
                fvals[-1] = fr
            else:
                if fr < fvals[-1]:
                    xc = centroid + rho * (xr - centroid)
                    fc = func(xc)
                    fev += 1
                    if fc <= fr:
                        simplex[-1] = xc
                        fvals[-1] = fc
                    else:
                        break
                else:
                    xc = centroid - rho * (simplex[-1] - centroid)
                    fc = func(xc)
                    fev += 1
                    if fc < fvals[-1]:
                        simplex[-1] = xc
                        fvals[-1] = fc
                    else:
                        break
            
            if fr >= fvals[-1]:
                for i in range(1, n+1):
                    simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                    fvals[i] = func(simplex[i])
                    fev += 1
        
        idx = np.argmin(fvals)
        return simplex[idx], fvals[idx]

    # Phase 1: Coarse optimization
    best_params = init_params.copy()
    best_coarse = evaluate_path_coarse(params_to_path(best_params))
    
    result_params, result_val = nelder_mead(objective, init_params, maxfev=2000)
    if -result_val > best_coarse:
        best_params = result_params
        best_coarse = -result_val
    
    # Phase 2: Local refinement with full evaluation
    def objective_full(params):
        if time.time() - start_time > time_limit:
            return 0.0
        path = params_to_path(params)
        return -evaluate_path_full(path)
    
    best_params, best_full_val = nelder_mead(objective_full, best_params, maxfev=1000)
    best_val = -best_full_val
    
    # Phase 3: Try a few perturbations
    for trial in range(3):
        if time.time() - start_time > time_limit - 60:
            break
        perturbed = best_params + np.random.normal(0, 0.05, size=len(best_params))
        p_params, p_val = nelder_mead(objective_full, perturbed, maxfev=500)
        if -p_val > best_val:
            best_params = p_params
            best_val = -p_val
    
    # Final evaluation
    final_path = params_to_path(best_params)
    final_val = evaluate_path_full(final_path)
    
    if final_val > best_val:
        best_val = final_val
    
    return final_path
