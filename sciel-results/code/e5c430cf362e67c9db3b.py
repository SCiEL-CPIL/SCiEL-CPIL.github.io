import numpy as np
import time
import math

def moving_sofa_2d() -> np.ndarray:
    start_time = time.time()
    deadline = 610.0  # Safety margin before 630s kill
    
    # Grid for evaluation, matching the evaluator
    # i = 0..449 -> x = -3 + 0.01i
    # j = 0..299 -> y = -1.5 + 0.01j
    xs = np.arange(-3, 1.5, 0.01)
    ys = np.arange(-1.5, 1.5, 0.01)
    X, Y = np.meshgrid(xs, ys, indexing='ij')
    P = np.column_stack((X.ravel(), Y.ravel()))
    n_points = P.shape[0]
    
    # Precompute base mask conditions? No, u,v depend on path.
    # But we can precompute the grid coordinates once.
    
    # Strategy:
    # Use a dense path: angles 1, 2, ..., 89.
    # This means m=1 for all interpolation steps in the evaluator.
    # We optimize x[1..89], y[1..89].
    # Total 178 variables.
    # Objective: Maximize count of points in intersection.
    
    angles = np.arange(1, 90)  # 1 to 89
    n_ctrl = len(angles)
    
    # Initial guess:
    # The optimal path for the moving sofa problem in a unit corridor
    # is known to be related to the Malfatti curves or similar.
    # A good initial guess is a smooth curve from (0,0) to (-2,0).
    # x(t) goes from 0 to -2.
    # y(t) goes from 0 up to some height and back to 0.
    # Max height is limited by corridor width.
    # Let's start with a semi-circle or sine wave.
    
    t = angles / 90.0
    x0 = -2.0 * t
    # Amplitude 1.0 seems reasonable for a unit width corridor.
    y0 = 1.0 * np.sin(np.pi * t)
    
    params0 = np.concatenate([x0, y0])
    
    def evaluate_path_fast(path_xy):
        """
        path_xy: array of shape (n_ctrl, 2) for angles 1..89
        Returns: number of valid points * 0.0001
        """
        # Full path includes endpoints (0,0,0) and (-2,0,90)
        # But we only need to check intersection of ALL placements.
        # The endpoints are fixed.
        # We can pre-compute the valid mask for endpoints?
        # No, because the rotation is different.
        
        # Let's build the list of all placements to check.
        # 0: (0, 0, 0)
        # 1..89: (x_i, y_i, i)
        # 90: (-2, 0, 90)
        
        valid = np.ones(n_points, dtype=bool)
        
        # Check endpoint 0
        # a=0, x=0, y=0
        # u = dx, v = dy
        dx = P[:, 0]
        dy = P[:, 1]
        u = dx
        v = dy
        in_base = ((u >= 0) & (u <= 1) & (v >= -3) & (v <= 1)) | \
                  ((u >= -3) & (u <= 1) & (v >= 0) & (v <= 1))
        valid = valid & in_base
        
        if not np.any(valid):
            return 0.0

        # Check intermediate points
        for i in range(n_ctrl):
            px = path_xy[i, 0]
            py = path_xy[i, 1]
            pa = angles[i]
            
            a_rad = np.deg2rad(pa)
            cos_a = np.cos(a_rad)
            sin_a = np.sin(a_rad)
            
            dx = P[:, 0] - px
            dy = P[:, 1] - py
            
            u = cos_a * dx + sin_a * dy
            v = -sin_a * dx + cos_a * dy
            
            in_base = ((u >= 0) & (u <= 1) & (v >= -3) & (v <= 1)) | \
                      ((u >= -3) & (u <= 1) & (v >= 0) & (v <= 1))
            
            valid = valid & in_base
            
            if not np.any(valid):
                return 0.0

        # Check endpoint 90
        # a=90, x=-2, y=0
        # R(-90) = [[0, 1], [-1, 0]]
        # u = 0*dx + 1*dy = dy
        # v = -1*dx + 0*dy = -dx
        dx = P[:, 0] - (-2.0)
        dy = P[:, 1] - 0.0
        
        u = dy
        v = -dx
        
        in_base = ((u >= 0) & (u <= 1) & (v >= -3) & (v <= 1)) | \
                  ((u >= -3) & (u <= 1) & (v >= 0) & (v <= 1))
        valid = valid & in_base
        
        if not np.any(valid):
            return 0.0

        return np.sum(valid) * 0.0001

    def objective(params):
        if time.time() > deadline - 5:
            return 0.0
        path_xy = params.reshape(-1, 2)
        return -evaluate_path_fast(path_xy)
    
    # Gradient Descent with Numerical Gradients
    # This is expensive, but we can use a coarser grid for gradient estimation
    # or just use a very small step size and many iterations.
    # Given 178 variables, full gradient is 178 evaluations per step.
    # Too slow for 600s if we do many steps.
    
    # Alternative: Coordinate Descent or Random Gradient Descent (SGD-like)
    # Or, use a smaller number of control points for optimization, 
    # but enforce smoothness via regularization?
    
    # Let's try a hybrid:
    # 1. Optimize a sparse set of points (e.g., every 5 degrees) using Nelder-Mead or L-BFGS-B.
    # 2. Interpolate to get dense path.
    # 3. Fine-tune the dense path using local perturbations (simulated annealing or random restarts)
    #    but only on the "critical" points?
    
    # Actually, the previous program used Nelder-Mead on 17 points (34 vars) and got 1.31.
    # Let's try L-BFGS-B on 17 points, which is better for smooth functions.
    # The function is non-smooth (count of points), but locally linear-ish.
    
    # Let's try optimizing 18 points: 5, 10, ..., 90? No, 90 is fixed.
    # 5, 10, ..., 85. (17 points).
    
    angles_sparse = np.arange(5, 90, 5)
    n_sparse = len(angles_sparse)
    
    t_sparse = angles_sparse / 90.0
    x0_sparse = -2.0 * t_sparse
    y0_sparse = 1.0 * np.sin(np.pi * t_sparse)
    params0_sparse = np.concatenate([x0_sparse, y0_sparse])
    
    def objective_sparse(params):
        if time.time() > deadline - 5:
            return 0.0
        path_xy = params.reshape(-1, 2)
        # Construct full dense path for evaluation?
        # The evaluator uses linear interpolation between our provided points.
        # If we provide points at 5, 10, ..., the evaluator adds intermediates.
        # So we MUST evaluate with the interpolated points.
        
        # Build full path
        full_path = np.zeros((n_sparse + 2, 3))
        full_path[0] = [0, 0, 0]
        for i in range(n_sparse):
            full_path[i+1, 0] = path_xy[i, 0]
            full_path[i+1, 1] = path_xy[i, 1]
            full_path[i+1, 2] = angles_sparse[i]
        full_path[-1] = [-2, 0, 90]
        
        # Densify
        dense_path = [full_path[0]]
        for i in range(len(full_path) - 1):
            p1 = full_path[i]
            p2 = full_path[i+1]
            da = abs(p2[2] - p1[2])
            m = max(1, int(np.ceil(da)))
            if m == 1:
                dense_path.append(p2)
            else:
                for k in range(1, m+1):
                    t = k / m
                    interp = p1 + t * (p2 - p1)
                    dense_path.append(interp)
        
        valid = np.ones(n_points, dtype=bool)
        
        for px, py, pa in dense_path:
            a_rad = np.deg2rad(pa)
            cos_a = np.cos(a_rad)
            sin_a = np.sin(a_rad)
            
            dx = P[:, 0] - px
            dy = P[:, 1] - py
            
            u = cos_a * dx + sin_a * dy
            v = -sin_a * dx + cos_a * dy
            
            in_base = ((u >= 0) & (u <= 1) & (v >= -3) & (v <= 1)) | \
                      ((u >= -3) & (u <= 1) & (v >= 0) & (v <= 1))
            
            valid = valid & in_base
            
            if not np.any(valid):
                return 0.0
                
        return -np.sum(valid) * 0.0001

    from scipy.optimize import minimize
    
    best_score = -np.inf
    best_params = params0_sparse
    
    # Try L-BFGS-B
    try:
        result = minimize(
            objective_sparse, 
            params0_sparse, 
            method='L-BFGS-B', 
            options={'maxiter': 100, 'ftol': 1e-6, 'gtol': 1e-5, 'disp': False}
        )
        if -result.fun > best_score:
            best_score = -result.fun
            best_params = result.x
    except Exception:
        pass
        
    # Try Nelder-Mead as fallback or refinement
    if time.time() < deadline - 30:
        try:
            result_nm = minimize(
                objective_sparse, 
                best_params, 
                method='Nelder-Mead', 
                options={'maxiter': 200, 'xatol': 1e-4, 'fatol': 1e-5, 'disp': False}
            )
            if -result_nm.fun > best_score:
                best_score = -result_nm.fun
                best_params = result_nm.x
        except Exception:
            pass

    # Final Step: 
    # The sparse optimization gives us a good path.
    # Can we improve by adding more points?
    # If we have time, we can take the sparse path, interpolate it to dense,
    # and then do a few steps of local search on the dense path?
    # Or simply return the sparse path?
    # The evaluator will interpolate the sparse path anyway.
    # So returning the sparse path is valid.
    
    # Construct final path from best_params
    final_path = np.zeros((n_sparse, 3))
    for i in range(n_sparse):
        final_path[i, 0] = best_params[2*i]
        final_path[i, 1] = best_params[2*i+1]
        final_path[i, 2] = angles_sparse[i]
        
    return final_path

