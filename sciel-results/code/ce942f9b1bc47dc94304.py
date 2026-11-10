# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import differential_evolution, minimize

def kakeya_needle_3d8() -> np.ndarray:
    N = 8
    deadline = time.time() + 590.0  # Safety margin before 630s kill
    
    # Precompute shear rates
    i_vals = np.arange(N, dtype=np.float64)
    shear = i_vals / N  # shape (N,)
    width = 1.0 / N

    def compute_1d_union_len(pos, c):
        """
        Compute 1D union length of intervals [pos_i + c*shear_i, pos_i + c*shear_i + width]
        at height c.
        pos: array of shape (N,)
        c: scalar height
        """
        L = pos + c * shear
        R = L + width
        order = np.argsort(L)
        Ls = L[order]
        Rs = R[order]
        
        # Standard interval union algorithm
        if N == 0:
            return 0.0
        total = 0.0
        cur_start = Ls[0]
        cur_end = Rs[0]
        for k in range(1, N):
            if Ls[k] > cur_end:
                total += cur_end - cur_start
                cur_start = Ls[k]
                cur_end = Rs[k]
            else:
                cur_end = max(cur_end, Rs[k])
        total += cur_end - cur_start
        return total

    def exact_volume(params):
        """
        Compute exact volume V = integral_0^1 Ux(c) * Uy(c) dc.
        params: array of shape (2*N,) where first N are x positions, next N are y positions.
        """
        x = params[:N]
        y = params[N:]
        
        # Find all breakpoints in [0,1] where the union structure changes.
        breakpoints = {0.0, 1.0}
        
        # Generate breakpoints for x-dimension
        for i in range(N):
            for j in range(N):
                if i == j:
                    continue
                di = shear[i] - shear[j]
                if abs(di) < 1e-15:
                    continue
                # L_i = L_j
                c1 = (x[j] - x[i]) / di
                if 0.0 <= c1 <= 1.0:
                    breakpoints.add(c1)
                
                # L_i = R_j
                c2 = (x[j] - x[i] + width) / di
                if 0.0 <= c2 <= 1.0:
                    breakpoints.add(c2)
                
                # R_i = L_j
                c3 = (x[j] - x[i] - width) / di
                if 0.0 <= c3 <= 1.0:
                    breakpoints.add(c3)

        # Generate breakpoints for y-dimension
        for i in range(N):
            for j in range(N):
                if i == j:
                    continue
                di = shear[i] - shear[j]
                if abs(di) < 1e-15:
                    continue
                c1 = (y[j] - y[i]) / di
                if 0.0 <= c1 <= 1.0:
                    breakpoints.add(c1)
                
                c2 = (y[j] - y[i] + width) / di
                if 0.0 <= c2 <= 1.0:
                    breakpoints.add(c2)
                
                c3 = (y[j] - y[i] - width) / di
                if 0.0 <= c3 <= 1.0:
                    breakpoints.add(c3)
        
        pts = sorted(breakpoints)
        
        # Integrate Ux(c) * Uy(c) over each subinterval.
        V = 0.0
        for m in range(len(pts) - 1):
            c0 = pts[m]
            c1 = pts[m+1]
            if c1 - c0 < 1e-12:
                continue
            
            u0_x = compute_1d_union_len(x, c0)
            u1_x = compute_1d_union_len(x, c1)
            u0_y = compute_1d_union_len(y, c0)
            u1_y = compute_1d_union_len(y, c1)
            
            dx = u1_x - u0_x
            dy = u1_y - u0_y
            integral_t = u0_x * u0_y + 0.5 * (u0_x * dy + u0_y * dx) + (dx * dy) / 3.0
            
            V += integral_t * (c1 - c0)
        
        return V

    # Initial guesses
    k = np.arange(N, dtype=np.float64)
    init_x = -k / (2.0 * N)
    init_y = -k / (2.0 * N)
    
    # Try a few different initial configurations
    init_candidates = [
        np.concatenate([init_x, init_y]),
        np.concatenate([init_x * 0.5, init_y * 0.5]),
        np.concatenate([init_x * 1.5, init_y * 1.5]),
        np.concatenate([np.linspace(-0.5, 0.5, N), np.linspace(-0.5, 0.5, N)]),
        # Add more diverse starts
        np.concatenate([init_x + 0.1*np.sin(k), init_y + 0.1*np.cos(k)]),
        np.concatenate([init_x - 0.1*np.cos(k), init_y - 0.1*np.sin(k)]),
    ]
    
    best_params = init_candidates[0].copy()
    best_val = exact_volume(best_params)
    
    # Evaluate all initial candidates
    for cand in init_candidates[1:]:
        val = exact_volume(cand)
        if val < best_val:
            best_val = val
            best_params = cand.copy()
    
    # Global optimization using differential evolution
    bounds = [(-2.0, 2.0)] * (2 * N)
    
    if time.time() < deadline - 150:
        try:
            result = differential_evolution(
                exact_volume, bounds, seed=42, maxiter=200, 
                tol=1e-10, popsize=20, polish=False,
                updating='immediate'
            )
            if np.all(np.isfinite(result.x)) and result.fun < best_val:
                best_params = result.x.copy()
                best_val = result.fun
        except Exception:
            pass
    
    # Local refinement with Nelder-Mead
    if time.time() < deadline - 60:
        try:
            result2 = minimize(exact_volume, best_params, method='Nelder-Mead',
                              options={'maxiter': 5000, 'xatol': 1e-12, 'fatol': 1e-14})
            if np.all(np.isfinite(result2.x)) and result2.fun < best_val:
                best_params = result2.x.copy()
                best_val = result2.fun
        except Exception:
            pass
    
    # Try L-BFGS-B for smoother convergence
    if time.time() < deadline - 40:
        try:
            result3 = minimize(exact_volume, best_params, method='L-BFGS-B',
                              options={'maxiter': 1000, 'ftol': 1e-15, 'gtol': 1e-14})
            if np.all(np.isfinite(result3.x)) and result3.fun < best_val:
                best_params = result3.x.copy()
                best_val = result3.fun
        except Exception:
            pass
    
    # Final polish with tighter tolerance
    if time.time() < deadline - 20:
        try:
            result4 = minimize(exact_volume, best_params, method='Nelder-Mead',
                              options={'maxiter': 10000, 'xatol': 1e-15, 'fatol': 1e-17})
            if np.all(np.isfinite(result4.x)) and result4.fun < best_val:
                best_params = result4.x.copy()
                best_val = result4.fun
        except Exception:
            pass

    # Construct the 64x2 array
    x_best = best_params[:N]
    y_best = best_params[N:]
    
    positions = np.zeros((64, 2), dtype=np.float64)
    for k_idx in range(64):
        i = k_idx // N
        j = k_idx % N
        positions[k_idx, 0] = x_best[i]
        positions[k_idx, 1] = y_best[j]
    
    return positions

# EVOLVE-BLOCK-END
