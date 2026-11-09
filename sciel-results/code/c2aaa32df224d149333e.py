import numpy as np
import time

def kakeya_needle_3d8() -> np.ndarray:
    N = 8
    w = 1.0 / N  # tube width = 0.125
    start_time = time.time()
    deadline = 630.0  # evaluator kills at 630s
    
    # Precompute grids
    i_grid, j_grid = np.meshgrid(np.arange(N), np.arange(N), indexing="ij")
    i_flat = i_grid.flatten().astype(float)
    j_flat = j_grid.flatten().astype(float)
    i_over_N = i_flat / N
    j_over_N = j_flat / N
    
    # Precompute fixed random points for consistent evaluation
    # We'll use the same seed but generate points on-the-fly for speed
    
    def compute_volume(xk, yk, num_samples=20000, seed=12345):
        """Fast volume estimator for local search. xk, yk are shape (64,)."""
        X0 = np.min(xk)
        X1 = np.max(xk + i_over_N)
        Y0 = np.min(yk)
        Y1 = np.max(yk + j_over_N)
        
        area = (X1 - X0) * (Y1 - Y0)
        if area <= 1e-9:
            return float('inf')
        
        rng = np.random.default_rng(seed)
        a = rng.uniform(X0, X1, num_samples)
        b = rng.uniform(Y0, Y1, num_samples)
        c = rng.uniform(0, 1, num_samples)
        
        # Vectorized check - process in chunks to limit memory
        chunk_size = 50000
        in_any_count = 0
        for start in range(0, num_samples, chunk_size):
            end = min(start + chunk_size, num_samples)
            a_c = a[start:end, np.newaxis]
            b_c = b[start:end, np.newaxis]
            c_c = c[start:end, np.newaxis]
            
            x_left = xk[np.newaxis, :] + c_c * i_over_N[np.newaxis, :]
            y_left = yk[np.newaxis, :] + c_c * j_over_N[np.newaxis, :]
            
            in_x = (a_c >= x_left) & (a_c <= x_left + w)
            in_y = (b_c >= y_left) & (b_c <= y_left + w)
            in_any_count += np.sum(np.any(in_x & in_y, axis=1))
        
        fraction = in_any_count / num_samples
        volume = fraction * area
        return volume
    
    def compute_volume_full(xk, yk, num_samples=500000):
        """Full precision volume estimator for final evaluation."""
        X0 = np.min(xk)
        X1 = np.max(xk + i_over_N)
        Y0 = np.min(yk)
        Y1 = np.max(yk + j_over_N)
        
        area = (X1 - X0) * (Y1 - Y0)
        if area <= 1e-9:
            return float('inf')
        
        rng = np.random.default_rng(12345)
        a = rng.uniform(X0, X1, num_samples)
        b = rng.uniform(Y0, Y1, num_samples)
        c = rng.uniform(0, 1, num_samples)
        
        c_col = c[:, np.newaxis]
        x_left = xk[np.newaxis, :] + c_col * i_over_N[np.newaxis, :]
        y_left = yk[np.newaxis, :] + c_col * j_over_N[np.newaxis, :]
        
        in_x = (a[:, np.newaxis] >= x_left) & (a[:, np.newaxis] <= x_left + w)
        in_y = (b[:, np.newaxis] >= y_left) & (b[:, np.newaxis] <= y_left + w)
        in_any = np.any(in_x & in_y, axis=1)
        
        fraction = np.mean(in_any)
        volume = fraction * area
        return volume
    
    # Parametric form with 12 parameters:
    # x_k = a0 + a1*i + a2*j + a3*i*j + a4*i^2 + a5*j^2
    # y_k = b0 + b1*i + b2*j + b3*i*j + b4*i^2 + b5*j^2
    
    def params_to_positions(params):
        a0, a1, a2, a3, a4, a5 = params[0:6]
        b0, b1, b2, b3, b4, b5 = params[6:12]
        
        xk = a0 + a1 * i_flat + a2 * j_flat + a3 * i_flat * j_flat + a4 * i_flat**2 + a5 * j_flat**2
        yk = b0 + b1 * i_flat + b2 * j_flat + b3 * i_flat * j_flat + b4 * i_flat**2 + b5 * j_flat**2
        return xk, yk
    
    def compute_volume_from_params(params, num_samples=20000):
        xk, yk = params_to_positions(params)
        return compute_volume(xk, yk, num_samples)
    
    # Multiple initial configurations
    alpha_init = 0.0625
    candidates = [
        # Symmetric linear
        np.array([0.0, -alpha_init, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -alpha_init, 0.0, 0.0, 0.0]),
        # With cross term
        np.array([0.0, -alpha_init, -alpha_init, 0.0, 0.0, 0.0, 0.0, -alpha_init, -alpha_init, 0.0, 0.0, 0.0]),
        # Pure x depends on i, y depends on j with different scales
        np.array([0.0, -0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -0.05, 0.0, 0.0, 0.0]),
        np.array([0.0, -0.05, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -0.1, 0.0, 0.0, 0.0]),
        # Quadratic terms
        np.array([0.0, -0.08, 0.0, 0.0, 0.01, 0.0, 0.0, 0.0, -0.08, 0.0, 0.0, 0.01]),
        np.array([0.0, -0.08, 0.0, 0.0, 0.0, 0.01, 0.0, 0.0, -0.08, 0.01, 0.0, 0.0]),
        # More aggressive linear
        np.array([0.0, -0.12, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -0.12, 0.0, 0.0, 0.0]),
        # Asymmetric
        np.array([0.0, -0.15, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -0.08, 0.0, 0.0, 0.0]),
    ]
    
    # Evaluate all candidates
    best_vol = float('inf')
    best_params = candidates[0].copy()
    for p in candidates:
        vol = compute_volume_from_params(p, 20000)
        if vol < best_vol:
            best_vol = vol
            best_params = p.copy()
    
    # Powell optimization (better than Nelder-Mead for this smooth parametric form)
    from scipy.optimize import minimize
    
    def objective(params):
        return compute_volume_from_params(params, 20000)
    
    # Powell optimization with tight tolerances
    result = minimize(objective, best_params, method='Powell', 
                      options={'maxiter': 3000, 'ftol': 1e-12, 'xtol': 1e-10, 'maxfev': 5000})
    
    if result.fun < best_vol:
        best_params = result.x
        best_vol = result.fun
    
    # Second round: Nelder-Mead from Powell result for local refinement
    result2 = minimize(objective, best_params, method='Nelder-Mead',
                       options={'maxiter': 2000, 'xatol': 1e-9, 'fatol': 1e-12, 'adaptive': True})
    
    if result2.fun < best_vol:
        best_params = result2.x
        best_vol = result2.fun
    
    # Check time
    elapsed = time.time() - start_time
    if elapsed > deadline - 30:
        xk, yk = params_to_positions(best_params)
        return np.stack([xk, yk], axis=1)
    
    # Refinement: use the parametric result as initial guess for full 128-DOF search
    xk, yk = params_to_positions(best_params)
    best_vol = compute_volume(xk, yk, 20000)
    
    # Coordinate descent with adaptive step size - more thorough
    step_size = 0.003
    no_improve_count = 0
    max_no_improve = 30
    min_step = 1e-8
    
    while time.time() - start_time < deadline - 60:
        improved = False
        for k in range(64):
            if time.time() - start_time > deadline - 60:
                break
            
            # Try 4 directions (±x, ±y) with current step size
            best_local_vol = best_vol
            best_local_dx = 0
            best_local_dy = 0
            
            for dx, dy in [(-step_size, 0), (step_size, 0), (0, -step_size), (0, step_size)]:
                xk_new = xk.copy()
                yk_new = yk.copy()
                xk_new[k] += dx
                yk_new[k] += dy
                
                vol = compute_volume(xk_new, yk_new, 10000)
                if vol < best_local_vol - 1e-12:
                    best_local_vol = vol
                    best_local_dx = dx
                    best_local_dy = dy
            
            if best_local_dx != 0 or best_local_dy != 0:
                xk[k] += best_local_dx
                yk[k] += best_local_dy
                best_vol = best_local_vol
                improved = True
                no_improve_count = 0
            else:
                no_improve_count += 1
        
        if not improved:
            step_size *= 0.6
            no_improve_count = 0
            if step_size < min_step:
                break
        else:
            no_improve_count = 0
    
    # Final high-precision polish
    if time.time() - start_time < deadline - 30:
        step_size = 1e-5
        for polish_round in range(5):
            if time.time() - start_time > deadline - 30:
                break
            for k in range(64):
                if time.time() - start_time > deadline - 30:
                    break
                best_local_vol = best_vol
                best_local_dx = 0
                best_local_dy = 0
                
                for dx, dy in [(-step_size, 0), (step_size, 0), (0, -step_size), (0, step_size),
                               (-step_size, -step_size), (step_size, step_size),
                               (-step_size, step_size), (step_size, -step_size)]:
                    xk_new = xk.copy()
                    yk_new = yk.copy()
                    xk_new[k] += dx
                    yk_new[k] += dy
                    
                    vol = compute_volume(xk_new, yk_new, 10000)
                    if vol < best_local_vol - 1e-14:
                        best_local_vol = vol
                        best_local_dx = dx
                        best_local_dy = dy
                
                if best_local_dx != 0 or best_local_dy != 0:
                    xk[k] += best_local_dx
                    yk[k] += best_local_dy
                    best_vol = best_local_vol
            
            step_size *= 0.5
    
    # Return best configuration
    positions = np.stack([xk, yk], axis=1)
    return positions
