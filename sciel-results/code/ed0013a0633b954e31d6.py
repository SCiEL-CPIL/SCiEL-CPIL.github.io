# EVOLVE-BLOCK-START
import numpy as np
import time

def moving_sofa_2d() -> np.ndarray:
    # Grid definition
    x_grid = np.arange(-3.0, 1.5, 0.01)
    y_grid = np.arange(-1.5, 1.5, 0.01)
    
    # Create meshgrid
    X, Y = np.meshgrid(x_grid, y_grid, indexing='ij')
    
    # Flatten for easier vectorization
    X_flat = X.flatten()
    Y_flat = Y.flatten()
    
    def check_inside(x, y, a_deg):
        theta = -np.deg2rad(a_deg)
        c = np.cos(theta)
        s = np.sin(theta)
        
        wx = X_flat - x
        wy = Y_flat - y
        
        u = wx * c - wy * s
        v = wx * s + wy * c
        
        in_r1 = (u >= 0.0) & (u <= 1.0) & (v >= -3.0) & (v <= 1.0)
        in_r2 = (u >= -3.0) & (u <= 1.0) & (v >= 0.0) & (v <= 1.0)
        
        inside = in_r1 | in_r2
        return inside

    def evaluate_path(path):
        full_path = np.vstack([
            [[0.0, 0.0, 0.0]],
            path,
            [[-2.0, 0.0, 90.0]]
        ])
        
        mask = np.ones(len(X_flat), dtype=bool)
        
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
                
                inside = check_inside(x, y, a)
                mask &= inside
                
                if not mask.any():
                    return 0.0

        return np.sum(mask) * 0.0001

    # Optimization
    # We'll optimize 8 control points (4 x, 4 y) at fixed angles 20, 40, 60, 80.
    
    def make_path_from_params(params):
        p1 = [params[0], params[1], 20.0]
        p2 = [params[2], params[3], 40.0]
        p3 = [params[4], params[5], 60.0]
        p4 = [params[6], params[7], 80.0]
        return np.array([p1, p2, p3, p4], dtype=np.float64)
    
    # Initial params
    current_params = np.array([-0.2, 0.2, -0.3, 0.3, -0.55, 0.2, -0.65, 0.25])
    
    start_time = time.time()
    time_limit = 590.0
    
    best_params = current_params.copy()
    best_val = evaluate_path(make_path_from_params(best_params))
    
    # Nelder-Mead style optimization
    def nelder_mead(func, x0, args=(), xtol=1e-4, ftol=1e-4, maxfev=10000, alpha=1.0, gamma=2.0, rho=0.5, sigma=0.5, time_limit=None, start_time=None):
        n = len(x0)
        # Initialize simplex
        simplex = np.zeros((n+1, n))
        simplex[0] = x0
        for i in range(n):
            simplex[i+1] = x0
            simplex[i+1, i] = x0[i] + 0.05 * abs(x0[i]) if x0[i] != 0 else 0.05
        
        fvals = np.array([func(x) for x in simplex])
        fev = n + 1
        
        while maxfev > 0 and fev < maxfev:
            if time_limit is not None and start_time is not None:
                if time.time() - start_time > time_limit:
                    break
            
            # Sort simplex by fvals
            idx = np.argsort(fvals)
            simplex = simplex[idx]
            fvals = fvals[idx]
            
            # Check convergence
            if np.max(np.abs(fvals - fvals[0])) < ftol:
                break
            if np.max(np.abs(simplex[0] - simplex[1])) < xtol:
                break
            
            # Centroid
            centroid = np.mean(simplex[:-1], axis=0)
            
            # Reflection
            xr = centroid + alpha * (centroid - simplex[-1])
            fr = func(xr)
            fev += 1
            
            if fr < fvals[0]:
                # Expansion
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
                # Accept reflection
                simplex[-1] = xr
                fvals[-1] = fr
            else:
                # Contraction
                if fr < fvals[-1]:
                    # Outer contraction
                    xc = centroid + rho * (xr - centroid)
                    fc = func(xc)
                    fev += 1
                    
                    if fc <= fr:
                        simplex[-1] = xc
                        fvals[-1] = fc
                    else:
                        break # Shrink
                else:
                    # Inner contraction
                    xc = centroid - rho * (simplex[-1] - centroid)
                    fc = func(xc)
                    fev += 1
                    
                    if fc < fvals[-1]:
                        simplex[-1] = xc
                        fvals[-1] = fc
                    else:
                        break # Shrink
            
            if fr >= fvals[-1]:
                # Shrink
                for i in range(1, n+1):
                    simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                    fvals[i] = func(simplex[i])
                    fev += 1
        
        # Return best point
        idx = np.argmin(fvals)
        return simplex[idx], fvals[idx], fev

    def objective(params):
        path = make_path_from_params(params)
        return -evaluate_path(path) # Minimize negative score

    # Run Nelder-Mead
    best_params, best_val_neg, fev = nelder_mead(
        objective, 
        current_params, 
        xtol=1e-4, 
        ftol=1e-5, 
        maxfev=5000,
        time_limit=time_limit,
        start_time=start_time
    )
    best_val = -best_val_neg
    
    # If we have time left, do a few more local searches with different initializations
    if time.time() - start_time < time_limit - 10:
        # Try a few random restarts
        for _ in range(5):
            if time.time() - start_time > time_limit - 10:
                break
            
            # Random perturbation
            new_params = best_params + np.random.normal(0, 0.1, size=len(best_params))
            
            new_params, new_val_neg, _ = nelder_mead(
                objective, 
                new_params, 
                xtol=1e-4, 
                ftol=1e-5, 
                maxfev=1000,
                time_limit=time_limit,
                start_time=start_time
            )
            new_val = -new_val_neg
            
            if new_val > best_val:
                best_val = new_val
                best_params = new_params

    final_path = make_path_from_params(best_params)
    
    return final_path

# EVOLVE-BLOCK-END
