# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import differential_evolution, minimize, minimize_scalar

def kakeya_needle_3d8() -> np.ndarray:
    """
    Discrete 3-D Kakeya set: union of 64 sheared tubes (N = 8).
    
    Strategy: 
    1. Global search via Differential Evolution
    2. Fast local polish using coordinate descent with Brent's method
    3. High-precision final polish with more c-samples
    """
    N = 8
    start_time = time.time()
    deadline = start_time + 615.0  # Safety margin before 630s kill
    
    # c-samples for objective evaluation (medium precision for search)
    rng = np.random.default_rng(42)
    M_obj = 100_000
    c_samples = rng.uniform(0, 1, M_obj)
    
    i_vals = np.arange(N, dtype=np.float64)
    shear_coeffs = i_vals / N  # shape (N,)
    
    def compute_1d_union_lens(x, c_samples):
        """
        Computes U(c) for all c_samples.
        x: shape (N,) - base positions
        c_samples: shape (M,)
        Returns: shape (M,) union lengths
        """
        L = x[:, None] + c_samples[None, :] * shear_coeffs[:, None]
        R = L + 1.0 / N
        
        # Transpose to (M, N)
        L_t = L.T
        R_t = R.T
        
        # Sort intervals by start point
        order = np.argsort(L_t, axis=1, kind='mergesort')
        L_sorted = np.take_along_axis(L_t, order, axis=1)
        R_sorted = np.take_along_axis(R_t, order, axis=1)
        
        # Compute union length via sweep
        max_r_up_to = np.maximum.accumulate(R_sorted, axis=1)
        max_r_before = np.empty_like(R_sorted)
        max_r_before[:, 0] = -np.inf
        max_r_before[:, 1:] = max_r_up_to[:, :-1]
        
        effective_start = np.maximum(L_sorted, max_r_before)
        contributions = np.maximum(0, R_sorted - effective_start)
        union_lens = np.sum(contributions, axis=1)
        
        return union_lens

    def make_positions(x, y):
        """Convert 1D positions x, y (shape N,) to 64x2 array."""
        positions = np.zeros((64, 2), dtype=np.float64)
        for k in range(64):
            i = k // 8
            j = k % 8
            positions[k, 0] = x[i]
            positions[k, 1] = y[j]
        return positions

    def validate_positions(pos):
        """Check if positions are valid (finite, V > 0)."""
        if not np.all(np.isfinite(pos)):
            return False
        x_k = pos[:, 0]
        y_k = pos[:, 1]
        i_vals = np.arange(64) // 8
        j_vals = np.arange(64) % 8
        
        X0 = np.min(x_k)
        X1 = np.max(x_k + (i_vals + 1) / 8.0)
        Y0 = np.min(y_k)
        Y1 = np.max(y_k + (j_vals + 1) / 8.0)
        
        if (X1 - X0) * (Y1 - Y0) <= 1e-9:
            return False
        return True

    def objective_full(params, c_samples):
        """
        Objective: minimize integral_0^1 Ux(c) * Uy(c) dc
        params: shape (16,) = [x0..x7, y0..y7]
        """
        x = params[:N]
        y = params[N:]
        
        U_x = compute_1d_union_lens(x, c_samples)
        U_y = compute_1d_union_lens(y, c_samples)
        
        val = np.sum(U_x * U_y)
        if not np.isfinite(val):
            return 1e9
        return val

    # Initial guess: exploit symmetry - place tubes to minimize overlap
    # Optimal structure: x_i = -i/N + offset, y_j = -j/N + offset
    # This creates a "staircase" that minimizes union volume
    k = np.arange(N, dtype=np.float64)
    x_init = -k / N
    y_init = -k / N
    params_init = np.concatenate([x_init, y_init])
    
    # Differential Evolution bounds: allow positions in [-1.5, 0.5]
    bounds = [(-1.5, 0.5)] * (2 * N)
    
    best_params = params_init.copy()
    best_val = objective_full(params_init, c_samples)
    
    # Phase 1: Global search via Differential Evolution
    if time.time() < deadline - 200:
        try:
            de_result = differential_evolution(
                lambda p: objective_full(p, c_samples),
                bounds=bounds,
                maxiter=80,
                popsize=12,
                tol=1e-5,
                seed=42,
                init='sobol',
                polish=False,
                updating='immediate'
            )
            if de_result.success or de_result.fun < best_val:
                if np.isfinite(de_result.fun) and de_result.fun < best_val:
                    best_params = de_result.x.copy()
                    best_val = de_result.fun
        except Exception:
            pass
    
    # Phase 2: Coordinate descent with Brent's method (much faster than Nelder-Mead)
    if time.time() < deadline - 80:
        for _ in range(5):  # Multiple passes
            if time.time() > deadline - 40:
                break
            improved = False
            
            # Optimize x coordinates one at a time
            for i in range(N):
                if time.time() > deadline - 40:
                    break
                
                def obj_x_i(xi):
                    p = best_params.copy()
                    p[i] = xi
                    return objective_full(p, c_samples)
                
                x0 = best_params[i]
                # Use Brent's method for 1D optimization - much faster and more precise
                res = minimize_scalar(obj_x_i, bounds=(x0 - 0.5, x0 + 0.5), method='bounded',
                                     options={'maxiter': 100, 'xatol': 1e-10})
                if res.success and np.isfinite(res.fun) and res.fun < best_val - 1e-12:
                    best_params[i] = res.x
                    best_val = res.fun
                    improved = True
            
            # Optimize y coordinates one at a time
            for j in range(N):
                if time.time() > deadline - 40:
                    break
                
                def obj_y_j(yj):
                    p = best_params.copy()
                    p[N + j] = yj
                    return objective_full(p, c_samples)
                
                y0 = best_params[N + j]
                res = minimize_scalar(obj_y_j, bounds=(y0 - 0.5, y0 + 0.5), method='bounded',
                                     options={'maxiter': 100, 'xatol': 1e-10})
                if res.success and np.isfinite(res.fun) and res.fun < best_val - 1e-12:
                    best_params[N + j] = res.x
                    best_val = res.fun
                    improved = True
            
            if not improved:
                break
    
    # Phase 3: Final high-precision polish with more c-samples
    if time.time() < deadline - 20:
        # Use more c-samples for higher precision
        M_final = 300_000
        c_final = np.random.default_rng(42).uniform(0, 1, M_final)
        
        for _ in range(3):
            if time.time() > deadline - 10:
                break
            improved = False
            
            for i in range(N):
                if time.time() > deadline - 10:
                    break
                
                def obj_x_i_final(xi):
                    p = best_params.copy()
                    p[i] = xi
                    return objective_full(p, c_final)
                
                x0 = best_params[i]
                res = minimize_scalar(obj_x_i_final, bounds=(x0 - 0.1, x0 + 0.1), method='bounded',
                                     options={'maxiter': 200, 'xatol': 1e-12})
                if res.success and np.isfinite(res.fun) and res.fun < best_val - 1e-14:
                    best_params[i] = res.x
                    best_val = res.fun
                    improved = True
            
            for j in range(N):
                if time.time() > deadline - 10:
                    break
                
                def obj_y_j_final(yj):
                    p = best_params.copy()
                    p[N + j] = yj
                    return objective_full(p, c_final)
                
                y0 = best_params[N + j]
                res = minimize_scalar(obj_y_j_final, bounds=(y0 - 0.1, y0 + 0.1), method='bounded',
                                     options={'maxiter': 200, 'xatol': 1e-12})
                if res.success and np.isfinite(res.fun) and res.fun < best_val - 1e-14:
                    best_params[N + j] = res.x
                    best_val = res.fun
                    improved = True
            
            if not improved:
                break
    
    x_best = best_params[:N]
    y_best = best_params[N:]
    
    positions = make_positions(x_best, y_best)
    
    if not validate_positions(positions):
        # Fallback to centered alignment
        fallback_x = -k / (2.0 * N)
        fallback_y = fallback_x.copy()
        positions = make_positions(fallback_x, fallback_y)
    
    return positions

# EVOLVE-BLOCK-END
