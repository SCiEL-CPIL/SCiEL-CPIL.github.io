# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize, differential_evolution


def kakeya_needle_3d8() -> np.ndarray:
    """
    Three-dimensional discrete Kakeya needle problem with grid parameter N = 8.

    Strategy:
    1. Model base positions via a low-dimensional Fourier parametric deformation
       (structural change: reduces 16 free vars to ~12 coherent params).
    2. Compute exact 2D union area via Gauss-Legendre quadrature.
    3. Optimize using Differential Evolution (global) then Nelder-Mead (local).
    4. Final polish via coordinate descent on the Fourier coefficients.
    """
    N = 8
    start_time = time.time()
    deadline = start_time + 580.0  # safety margin before 600s kill

    # --- Fourier Parametric Model ---
    # base[i] = a0 + a1*cos(i*t) + b1*sin(i*t) + a2*cos(2*i*t) + b2*sin(2*i*t)
    # where t = 2*pi/N. We use 5 params per dimension (a0, a1, b1, a2, b2).
    # Total 10 params. This enforces smoothness and global coherence.
    
    # Initial values from Keich construction to guide the optimizer
    pos = int(np.log2(N))
    j_terms = np.arange(1, pos + 1)
    keich = np.empty(N, dtype=np.float64)
    for idx in range(N):
        eps = np.array(list(str(bin(idx))[2:]), dtype=int)
        eps = np.pad(eps, (pos - len(eps), 0), "constant")
        keich[idx] = np.sum((1 - j_terms) * eps * (2.0 ** -j_terms) / pos)

    # Fit initial Fourier coefficients to Keich
    def fit_fourier(data, N):
        i = np.arange(N)
        t = 2 * np.pi / N
        # Design matrix: [1, cos(i*t), sin(i*t), cos(2*i*t), sin(2*i*t)]
        A = np.column_stack([
            np.ones(N),
            np.cos(i * t),
            np.sin(i * t),
            np.cos(2 * i * t),
            np.sin(2 * i * t)
        ])
        # Least squares
        coeffs, _, _, _ = np.linalg.lstsq(A, data, rcond=None)
        return coeffs

    x_coeffs_init = fit_fourier(keich, N)
    y_coeffs_init = fit_fourier(keich, N)
    
    # We can also try a slightly different initial guess for y to break symmetry
    # but let's stick with symmetric for now, DE will explore.
    # Actually, let's add a small perturbation to y to help DE escape symmetry if needed?
    # No, DE is stochastic. Let's just use the fit.

    # --- Exact 2D Area Computation ---
    # Use 48-point Gauss-Legendre quadrature. Fast enough for DE.
    c_vals, w_vals = np.polynomial.legendre.leggauss(48)
    c_vals = 0.5 * (c_vals + 1.0)
    w_vals = 0.5 * w_vals
    idx_arange = np.arange(N, dtype=np.float64)
    interval_widths = (idx_arange + 1.0) / N

    def area_2d(base):
        """
        Compute area of union of N sheared parallelograms in (coord, c) plane.
        """
        total_area = 0.0
        # Vectorize over c? No, interval merging is sequential per c.
        # But N is small (8), so the inner loop is tiny.
        for c, w in zip(c_vals, w_vals):
            lefts = base + c * idx_arange / N
            rights = lefts + interval_widths
            
            order = np.argsort(lefts)
            lefts_sorted = lefts[order]
            rights_sorted = rights[order]
            
            total_len = 0.0
            cur_l = lefts_sorted[0]
            cur_r = rights_sorted[0]
            for k in range(1, N):
                if lefts_sorted[k] <= cur_r:
                    if rights_sorted[k] > cur_r:
                        cur_r = rights_sorted[k]
                else:
                    total_len += cur_r - cur_l
                    cur_l = lefts_sorted[k]
                    cur_r = rights_sorted[k]
            total_len += cur_r - cur_l
            
            total_area += w * total_len
        
        return total_area

    def coeffs_to_base(coeffs, N):
        i = np.arange(N)
        t = 2 * np.pi / N
        base = (coeffs[0] + 
                coeffs[1] * np.cos(i * t) + 
                coeffs[2] * np.sin(i * t) + 
                coeffs[3] * np.cos(2 * i * t) + 
                coeffs[4] * np.sin(2 * i * t))
        return base

    def objective(params):
        x_coeffs = params[:5]
        y_coeffs = params[5:]
        
        x_base = coeffs_to_base(x_coeffs, N)
        y_base = coeffs_to_base(y_coeffs, N)
        
        vx = area_2d(x_base)
        vy = area_2d(y_base)
        
        if vx <= 0 or vy <= 0:
            return 1e10
        return vx * vy

    # --- Optimization Phase 1: Differential Evolution ---
    # Bounds for Fourier coefficients. 
    # base[i] should be in a reasonable range, say [-1, 2].
    # Coeffs are roughly in [-1, 1] range. Let's use [-2, 2] for safety.
    bounds = [(-2.0, 2.0)] * 10
    init_params = np.concatenate([x_coeffs_init, y_coeffs_init])

    best_params = init_params.copy()
    best_obj = objective(best_params)

    if time.time() < deadline - 100.0:
        try:
            result = differential_evolution(
                objective,
                bounds,
                init='latinhypercube',
                seed=12345,
                maxiter=1000,
                tol=1e-6,
                popsize=15,
                polish=False,  # We will do our own polish
                update_interval=10,
                callback=lambda x, convergence: None
            )
            if result.fun < best_obj:
                best_obj = result.fun
                best_params = result.x
        except Exception:
            pass

    # --- Optimization Phase 2: Nelder-Mead ---
    if time.time() < deadline - 50.0:
        try:
            result = minimize(
                objective,
                best_params,
                method='Nelder-Mead',
                options={
                    'maxiter': 20000,
                    'xatol': 1e-8,
                    'fatol': 1e-10,
                    'adaptive': True,
                    'initial_simplex': 0.05 * np.ones(10)
                }
            )
            if result.fun < best_obj:
                best_obj = result.fun
                best_params = result.x
        except Exception:
            pass

    # --- Optimization Phase 3: Coordinate Descent on Coeffs ---
    if time.time() < deadline - 10.0:
        step = 1e-4
        for iteration in range(1000):
            if time.time() > deadline - 10.0:
                break
            improved = False
            for idx in range(10):
                old_val = best_params[idx]
                for delta in [step, -step, step/2, -step/2, step/4, -step/4]:
                    best_params[idx] = old_val + delta
                    obj = objective(best_params)
                    if obj < best_obj - 1e-14:
                        best_obj = obj
                        improved = True
                        break
                    best_params[idx] = old_val
                if not improved:
                    best_params[idx] = old_val
            if not improved:
                step *= 0.7
                if step < 1e-7:
                    break

    # --- Final Check: Try direct 16-var optimization if time permits ---
    # If the Fourier model is restrictive, direct optimization might be better.
    # But let's stick to the Fourier model as per the directive "structural change".
    # If we have lots of time left, we could try direct NM on 16 vars starting from Fourier result.
    
    if time.time() < deadline - 30.0:
        x_base_final = coeffs_to_base(best_params[:5], N)
        y_base_final = coeffs_to_base(best_params[5:], N)
        direct_params = np.concatenate([x_base_final, y_base_final])
        
        def objective_direct(params):
            vx = area_2d(params[:N])
            vy = area_2d(params[N:])
            if vx <= 0 or vy <= 0:
                return 1e10
            return vx * vy

        best_direct_obj = objective_direct(direct_params)
        if best_direct_obj < best_obj:
            best_obj = best_direct_obj
            best_params_direct = direct_params
        else:
            best_params_direct = direct_params

        # Run NM on direct params
        try:
            result = minimize(
                objective_direct,
                best_params_direct,
                method='Nelder-Mead',
                options={
                    'maxiter': 10000,
                    'xatol': 1e-8,
                    'fatol': 1e-10,
                    'adaptive': True,
                    'initial_simplex': 0.01 * np.ones(2 * N)
                }
            )
            if result.fun < best_obj:
                best_obj = result.fun
                best_params_direct = result.x
        except Exception:
            pass
        
        # Final coordinate descent on direct params
        if time.time() < deadline - 5.0:
            step = 1e-5
            for iteration in range(500):
                if time.time() > deadline - 5.0:
                    break
                improved = False
                for idx in range(2 * N):
                    old_val = best_params_direct[idx]
                    for delta in [step, -step]:
                        best_params_direct[idx] = old_val + delta
                        obj = objective_direct(best_params_direct)
                        if obj < best_obj - 1e-14:
                            best_obj = obj
                            improved = True
                            break
                        best_params_direct[idx] = old_val
                    if not improved:
                        best_params_direct[idx] = old_val
                if not improved:
                    step *= 0.7
                    if step < 1e-8:
                        break
            
            x_final = best_params_direct[:N]
            y_final = best_params_direct[N:]
        else:
            x_final = best_params_direct[:N]
            y_final = best_params_direct[N:]
    else:
        x_final = coeffs_to_base(best_params[:5], N)
        y_final = coeffs_to_base(best_params[5:], N)

    # --- Build final positions array ---
    i_grid, j_grid = np.meshgrid(np.arange(N), np.arange(N), indexing="ij")
    i_flat = i_grid.flatten()
    j_flat = j_grid.flatten()

    base_x = x_final[i_flat]
    base_y = y_final[j_flat]
    positions = np.stack([base_x, base_y], axis=1)
    
    # Validity check
    if not np.all(np.isfinite(positions)):
        # Fallback to Keich
        base_x = keich[i_flat]
        base_y = keich[j_flat]
        positions = np.stack([base_x, base_y], axis=1)

    return positions


# EVOLVE-BLOCK-END
