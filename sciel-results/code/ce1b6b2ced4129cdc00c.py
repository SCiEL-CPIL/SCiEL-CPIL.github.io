"""Improved growth-index ansatz with direct optimization from task structure."""
import numpy as np
from scipy.optimize import minimize
import time
import sys

# EVOLVE-BLOCK-START

# Deadline guard: evaluator kills after 90s; use 80s safety margin
_START_TIME = time.time()
_DEADLINE = _START_TIME + 55.0  # 55s limit to be safe, leaving time for import/setup

def _check_deadline():
    return time.time() > _DEADLINE

def construct():
    """Return six coefficients [c0..c5] for the growth-index ansatz.
    
    gamma = c0 + c1*q + c2*q^2 + c3*u + c4*wa*a + c5*u*q
    f = Om(a)^gamma
    D(a)/D(1) = exp(-integral_ln(a)^0 f(exp(x)) dx)
    
    We optimize these coefficients to minimize worst-case relative error
    across validation cosmologies for both f and D.
    """
    
    # Initial best result: the seed
    best_coeff = np.array([0.55, 0.0, 0.0, 0.0, 0.0, 0.0])
    best_score = np.inf
    
    # Representative validation cosmologies (Om0, w0, wa)
    # We choose a spread to cover the likely test space
    val_cosmos = [
        (0.30, -1.0, 0.0),   # Lambda CDM
        (0.30, -1.0, 0.1),   # Positive wa
        (0.30, -1.0, -0.1),  # Negative wa
        (0.30, -0.9, 0.0),   # w0 > -1
        (0.30, -1.1, 0.0),   # w0 < -1
        (0.28, -1.0, 0.0),   # Lower Om0
        (0.32, -1.0, 0.0),   # Higher Om0
    ]
    
    # Scale factor grid for evaluation (log-spaced)
    a_grid = np.logspace(np.log10(0.1), 0.0, 100)
    
    # Pre-compute reference values for each cosmology
    ref_data = []
    
    def _compute_reference(Om0, w0, wa, a_vals):
        """
        Compute reference f and D for a given cosmology and a-grid.
        Uses numerical integration of the growth ODE.
        """
        a_vals = np.asarray(a_vals)
        n = len(a_vals)
        
        # 1. Compute Omega_m(a) for all a
        Om_a = np.zeros(n)
        for i, a in enumerate(a_vals):
            if a <= 0:
                Om_a[i] = 1.0
                continue
            u = 1.0 + w0 + wa * (1.0 - a)
            if abs(wa) < 1e-10:
                if abs(1.0 + w0) < 1e-10:
                    rho_de_ratio = 1.0
                else:
                    rho_de_ratio = a ** (-3.0 * (1.0 + w0))
            else:
                exponent = -3.0 * np.log(a) * (1.0 + w0 + wa) + 3.0 * wa * (a - 1.0)
                rho_de_ratio = np.exp(exponent)
            
            rho_m_ratio = a ** -3.0
            denom = Om0 * rho_m_ratio + (1.0 - Om0) * rho_de_ratio
            if denom == 0:
                Om_a[i] = 0.0
            else:
                Om_a[i] = Om0 * rho_m_ratio / denom
        
        # 2. Solve ODE for D(a)
        x_grid = np.log(a_vals)
        
        def ode_func(x, y):
            a = np.exp(x)
            u = 1.0 + w0 + wa * (1.0 - a)
            if abs(wa) < 1e-10:
                if abs(1.0 + w0) < 1e-10:
                    rho_de_ratio = 1.0
                else:
                    rho_de_ratio = a ** (-3.0 * (1.0 + w0))
            else:
                exponent = -3.0 * np.log(a) * (1.0 + w0 + wa) + 3.0 * wa * (a - 1.0)
                rho_de_ratio = np.exp(exponent)
            
            rho_m_ratio = a ** -3.0
            denom = Om0 * rho_m_ratio + (1.0 - Om0) * rho_de_ratio
            if denom == 0:
                Om = 0.0
            else:
                Om = Om0 * rho_m_ratio / denom
            
            P = 3.0 + 1.5 * Om
            Q = 1.5 * Om
            return [y[1], -P * y[1] + Q * y[0]]
        
        x_min = np.log(0.1)
        y0 = [0.1, 1.0]
        
        try:
            sol = solve_ivp(ode_func, [x_min, 0.0], y0, t_eval=x_grid, method='RK45', rtol=1e-8, atol=1e-10)
            if not sol.success:
                return None
            D_vals = sol.y[0]
            D_prime_vals = sol.y[1]
            
            idx_1 = np.argmin(np.abs(x_grid))
            D_norm = D_vals[idx_1]
            if D_norm == 0:
                return None
            D_vals = D_vals / D_norm
            D_prime_vals = D_prime_vals / D_norm
            
            f_ref = D_prime_vals
            D_ref = D_vals
            
            return {
                'a': a_vals,
                'Om': Om_a,
                'f': f_ref,
                'D': D_ref,
                'w0': w0,
                'wa': wa
            }
        except Exception:
            return None

    from scipy.integrate import solve_ivp
    
    for c in val_cosmos:
        ref = _compute_reference(c[0], c[1], c[2], a_grid)
        if ref:
            ref_data.append(ref)
        if _check_deadline():
            break

    if not ref_data:
        return best_coeff.tolist()

    # Pre-compute arrays for fast evaluation
    a_arr = np.array([r['a'] for r in ref_data]).flatten()
    Om_arr = np.array([r['Om'] for r in ref_data]).flatten()
    f_ref_arr = np.array([r['f'] for r in ref_data]).flatten()
    D_ref_arr = np.array([r['D'] for r in ref_data]).flatten()
    w0_arr = np.array([r['w0'] for r in ref_data]).flatten()
    wa_arr = np.array([r['wa'] for r in ref_data]).flatten()
    x_arr = np.log(a_arr)
    
    # Pre-compute cumulative trapezoid weights for D calculation
    # We need integral from 0 to x for each point
    # Group by cosmology for proper integration
    n_cosmo = len(ref_data)
    n_pts = len(a_grid)
    
    # For each cosmology, compute the cumulative integral structure
    # I_k = cumulative_trapezoid(f_ours, x, initial=0) for cosmology k
    # This requires f_ours which depends on coeff, so we can't pre-compute fully
    # But we can pre-compute the trapezoid weights
    
    # For cumulative trapezoid: I[i] = sum_{j=1}^{i} 0.5*(f[j-1]+f[j])*(x[j]-x[j-1])
    # So I[i] = sum_{j=1}^{i} w_j * f[j-1] + w_j * f[j]
    # where w_j = 0.5*(x[j]-x[j-1])
    # This can be written as a matrix-vector product
    
    # Pre-compute the integration matrix for each cosmology
    # M_k such that I_k = M_k @ f_ours_k
    # For cumulative trapezoid with initial=0:
    # I[0] = 0
    # I[i] = I[i-1] + 0.5*(f[i-1]+f[i])*(x[i]-x[i-1])
    
    # Let's pre-compute dx and the cumulative structure
    dx = np.diff(x_arr.reshape(n_cosmo, n_pts), axis=1)  # shape (n_cosmo, n_pts-1)
    dx_half = 0.5 * dx  # shape (n_cosmo, n_pts-1)
    
    # For each cosmology, the cumulative integral can be computed as:
    # I[i] = sum_{j=1}^{i} dx_half[j-1] * (f[j-1] + f[j])
    # = sum_{j=1}^{i} dx_half[j-1]*f[j-1] + dx_half[j-1]*f[j]
    
    # We can build a matrix A such that I = A @ f
    # A[i, j] = coefficient of f[j] in I[i]
    
    # For efficiency, let's just compute it on the fly but vectorized
    
    def _compute_D_from_f(f_ours_flat):
        """Compute D from f using pre-computed structure.
        f_ours_flat: flat array of length n_cosmo * n_pts
        Returns: flat array of D values
        """
        f_reshaped = f_ours_flat.reshape(n_cosmo, n_pts)
        D = np.ones_like(f_reshaped)
        
        for k in range(n_cosmo):
            f_k = f_reshaped[k]
            x_k = x_arr.reshape(n_cosmo, n_pts)[k]
            
            # Cumulative trapezoid from 0 to x
            # x is decreasing (from ln(0.1) to 0)
            # We want integral from x to 0, which is -integral from 0 to x
            # cumulative_trapezoid gives integral from x[0] to x[i]
            # Since x is decreasing, the integral from x[0] to x[i] is negative
            # We want integral from x[i] to 0 = -integral from 0 to x[i]
            # = integral from x[i] to x[0] (since x[0] < 0 < x[i] is false, x[0] < x[i] < 0)
            # Actually x goes from ln(0.1) ≈ -2.3 to 0
            # integral from x[i] to 0 = integral from x[i] to x[n-1]
            # cumulative_trapezoid(f, x, initial=0) gives integral from x[0] to x[i]
            # So integral from x[i] to x[n-1] = total - integral from x[0] to x[i]
            
            # Simpler: just use cumulative_trapezoid and adjust
            from scipy.integrate import cumulative_trapezoid
            I = cumulative_trapezoid(f_k, x_k, initial=0)
            # I[i] = integral from x[0] to x[i]
            # We want integral from x[i] to x[n-1] = x[n-1] - x[0] integral... 
            # Actually: integral from x[i] to x[n-1] = integral from x[0] to x[n-1] - integral from x[0] to x[i]
            total_I = I[-1]
            I_from_xi_to_end = total_I - I
            # D = exp(-I_from_xi_to_end)
            D[k] = np.exp(-I_from_xi_to_end)
        
        return D.flatten()

    def objective(coeff):
        c0, c1, c2, c3, c4, c5 = coeff
        
        q = 1.0 - Om_arr
        u = 1.0 + w0_arr + wa_arr * (1.0 - a_arr)
        
        gamma = c0 + c1*q + c2*q**2 + c3*u + c4*wa_arr*a_arr + c5*u*q
        
        # Check bounds - use soft penalty for smoothness
        gamma_min = np.min(gamma)
        gamma_max = np.max(gamma)
        if gamma_min < 0.1 or gamma_max > 1.2:
            # Soft penalty
            pen = 0.0
            if gamma_min < 0.1:
                pen += 100.0 * (0.1 - gamma_min)**2
            if gamma_max > 1.2:
                pen += 100.0 * (gamma_max - 1.2)**2
            return pen + 1.0
        
        f_ours = np.power(np.maximum(Om_arr, 1e-15), gamma)
        D_ours = _compute_D_from_f(f_ours)
        
        # Relative errors
        mask_f = f_ref_arr > 1e-10
        mask_D = D_ref_arr > 1e-10
        
        err_f = np.zeros_like(f_ours)
        err_D = np.zeros_like(D_ours)
        
        if np.any(mask_f):
            err_f[mask_f] = np.abs((f_ours[mask_f] - f_ref_arr[mask_f]) / f_ref_arr[mask_f])
        
        if np.any(mask_D):
            err_D[mask_D] = np.abs((D_ours[mask_D] - D_ref_arr[mask_D]) / D_ref_arr[mask_D])
        
        # Use smooth max (log-sum-exp) for better gradient behavior
        # max_err ≈ logsumexp(log(err_f), log(err_D)) - log(2)
        # But we want the actual max, not a smooth approximation
        # For optimization, use a soft max with temperature
        all_errs = np.concatenate([err_f, err_D])
        all_errs = np.maximum(all_errs, 1e-15)
        
        # Temperature for soft max
        beta = 50.0
        log_errs = np.log(all_errs)
        max_log = np.max(log_errs)
        soft_max = max_log + np.log(np.mean(np.exp(beta * (log_errs - max_log))))
        # This approximates (1/beta) * log(sum(exp(beta * err))) which is close to max for large beta
        # Actually, let's use the proper soft max:
        # soft_max = (1/beta) * log(sum(exp(beta * err)))
        # For large beta, this approaches max(err)
        
        return soft_max

    # Optimization
    x0 = best_coeff
    
    # Phase 1: Nelder-Mead for global exploration
    try:
        res = minimize(objective, x0, method='Nelder-Mead', 
                      options={'maxiter': 300, 'xatol': 1e-5, 'fatol': 1e-5, 'adaptive': True})
        if res.fun < best_score:
            best_coeff = res.x
            best_score = res.fun
    except Exception:
        pass
    
    if _check_deadline():
        return best_coeff.tolist()
    
    # Phase 2: L-BFGS-B with smooth objective for refinement
    try:
        res2 = minimize(objective, best_coeff, method='L-BFGS-B', 
                       bounds=[(-10,10)]*6, 
                       options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-8})
        if res2.fun < best_score:
            best_coeff = res2.x
            best_score = res2.fun
    except Exception:
        pass
    
    if _check_deadline():
        return best_coeff.tolist()
    
    # Phase 3: Final polish with tighter tolerances
    try:
        res3 = minimize(objective, best_coeff, method='L-BFGS-B', 
                       bounds=[(-10,10)]*6, 
                       options={'maxiter': 300, 'ftol': 1e-15, 'gtol': 1e-10})
        if res3.fun < best_score:
            best_coeff = res3.x
            best_score = res3.fun
    except Exception:
        pass

    best_coeff = np.clip(best_coeff, -10.0, 10.0)
    
    return best_coeff.tolist()

# EVOLVE-BLOCK-END
