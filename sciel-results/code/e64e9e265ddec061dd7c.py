"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.integrate import solve_ivp
from scipy.optimize import minimize

# Global cache for reference data
_REF_CACHE = None

def _compute_om_a(a, Om0, w0, wa):
    """Compute matter density parameter Om(a) for CPL dark energy."""
    if a <= 0:
        return 1.0
    de_exponent = -3.0 * (1.0 + w0 + wa)
    if a >= 1.0:
        de_factor = 1.0
    else:
        one_minus_a = max(1.0 - a, 1e-15)
        de_factor = one_minus_a ** de_exponent
    E2 = Om0 / (a ** 3) + (1.0 - Om0) * de_factor
    Om_a = (Om0 / (a ** 3)) / E2
    return Om_a

def _compute_om_a_vec(a, Om0, w0, wa):
    """Vectorized compute matter density parameter Om(a) for CPL dark energy."""
    a = np.asarray(a)
    de_exponent = -3.0 * (1.0 + w0 + wa)
    de_factor = np.ones_like(a)
    mask = a < 1.0
    if np.any(mask):
        one_minus_a = np.maximum(1.0 - a[mask], 1e-15)
        de_factor[mask] = one_minus_a ** de_exponent
    E2 = Om0 / (a ** 3) + (1.0 - Om0) * de_factor
    Om_a = (Om0 / (a ** 3)) / E2
    return Om_a

def _solve_growth_ode(Om0, w0, wa, x_min, x_max, n_points=500):
    """
    Solve the GR growth ODE for f and D.
    Returns x, f, D_norm.
    """
    def ode_func(x, y):
        D, f = y
        a = np.exp(x)
        Om_a = _compute_om_a(a, Om0, w0, wa)
        D_prime = f * D
        f_prime = -f**2 - 1.5 * Om_a * f + 1.5 * Om_a
        return [D_prime, f_prime]
    
    y0 = [np.exp(x_min), 1.0]
    x_eval = np.linspace(x_min, x_max, n_points)
    
    sol = solve_ivp(ode_func, [x_min, x_max], y0, t_eval=x_eval, 
                    method='RK45', rtol=1e-10, atol=1e-12, max_step=0.01)
    
    if not sol.success:
        return None, None, None
    
    x_arr = sol.t
    D_arr = sol.y[0]
    f_arr = sol.y[1]
    
    D_norm = D_arr / D_arr[-1]
    return x_arr, D_norm, f_arr

def _init_cache():
    """Initialize reference data for validation cosmologies."""
    global _REF_CACHE
    if _REF_CACHE is not None:
        return
    
    # Validation cosmologies (disjoint public validation models)
    # We use a representative set covering the expected range
    test_models = [
        (0.3, -1.0, 0.0),
        (0.3, -0.9, -0.3),
        (0.3, -1.2, 0.5),
        (0.3, -0.8, 0.0),
        (0.2, -1.0, 0.0),
        (0.4, -1.0, 0.0),
        (0.3, -1.0, -1.0),
        (0.3, -1.0, 1.0),
        (0.3, -0.5, 0.0),
        (0.3, -1.5, 0.5),
    ]
    
    # Validation a values
    a_vals = np.linspace(0.05, 0.99, 50)
    x_vals = np.log(a_vals)
    
    # Fine grid for integration of D
    x_min_fine = np.log(0.01)
    x_max_fine = 0.0
    x_fine = np.linspace(x_min_fine, x_max_fine, 1000)
    
    cache = {
        'models': test_models,
        'a_vals': a_vals,
        'x_vals': x_vals,
        'x_fine': x_fine,
        'f_ref': [],
        'D_ref': [],
        'Om_ref': [],
        'q_ref': [],
        'u_ref': []
    }
    
    for Om0, w0, wa in test_models:
        # Solve ODE on fine grid to get f_ref on x_fine
        # And also get D_ref at a_vals
        # We can solve once on a grid that covers both, or interpolate.
        # Let's solve on x_fine directly if it covers the range.
        # x_fine is from ln(0.01) to 0. a_vals are 0.05 to 0.99.
        # So x_fine covers the range.
        
        x_arr, D_norm, f_arr = _solve_growth_ode(Om0, w0, wa, x_min_fine, x_max_fine, n_points=1000)
        
        if x_arr is None:
            continue
            
        # We need f and D at the specific a_vals (x_vals)
        f_at_a = np.interp(x_vals, x_arr, f_arr)
        D_at_a = np.interp(x_vals, x_arr, D_norm)
        
        # We also need f on the fine grid x_fine for integration of D_approx
        # x_arr is already x_fine (since we passed n_points=1000 and same range)
        # But to be safe, let's interpolate if grids don't match exactly
        f_on_fine = np.interp(x_fine, x_arr, f_arr)
        
        # Precompute Om, q, u at a_vals
        Om_a = _compute_om_a_vec(a_vals, Om0, w0, wa)
        q = 1.0 - Om_a
        u = 1.0 + w0 + wa * (1.0 - a_vals)
        
        cache['f_ref'].append(f_at_a)
        cache['D_ref'].append(D_at_a)
        cache['Om_ref'].append(Om_a)
        cache['q_ref'].append(q)
        cache['u_ref'].append(u)
        cache['f_on_fine'].append(f_on_fine)
        
    _REF_CACHE = cache

def _growth_index_error(coeffs):
    """Compute the worst relative error for a given set of 6 coefficients."""
    if _REF_CACHE is None:
        _init_cache()
        
    c0, c1, c2, c3, c4, c5 = coeffs
    max_rel_err = 0.0
    
    a_vals = _REF_CACHE['a_vals']
    x_fine = _REF_CACHE['x_fine']
    dx_fine = x_fine[1] - x_fine[0]
    
    for i, (Om0, w0, wa) in enumerate(_REF_CACHE['models']):
        Om_a = _REF_CACHE['Om_ref'][i]
        q = _REF_CACHE['q_ref'][i]
        u = _REF_CACHE['u_ref'][i]
        f_ref = _REF_CACHE['f_ref'][i]
        D_ref = _REF_CACHE['D_ref'][i]
        f_on_fine_ref = _REF_CACHE['f_on_fine'][i] # Not used for approx, but good for debug
        
        # Compute gamma
        gamma = c0 + c1 * q + c2 * q**2 + c3 * u + c4 * wa * a_vals + c5 * u * q
        
        # Check bounds
        if np.any(gamma < 0.1) or np.any(gamma > 1.2):
            return 10000.0
            
        # Compute f_approx
        f_approx = Om_a ** gamma
        
        # Relative error in f
        rel_err_f = np.abs(f_approx - f_ref) / np.maximum(np.abs(f_ref), 1e-10)
        max_rel_err = max(max_rel_err, np.max(rel_err_f))
        
        # Compute D_approx
        # We need integral of f_approx / a da from a to 1
        # = integral of f_approx dx from ln(a) to 0
        
        # Compute f_approx on x_fine
        # We need Om_a on x_fine
        a_fine = np.exp(x_fine)
        Om_fine = _compute_om_a_vec(a_fine, Om0, w0, wa)
        q_fine = 1.0 - Om_fine
        u_fine = 1.0 + w0 + wa * (1.0 - a_fine)
        
        gamma_fine = c0 + c1 * q_fine + c2 * q_fine**2 + c3 * u_fine + c4 * wa * a_fine + c5 * u_fine * q_fine
        
        # Clip gamma to valid range to avoid numerical issues in exp/pow if slightly out
        # But we already checked bounds at a_vals. The fine grid might go out of bounds?
        # The task says "Gamma must lie in [0.1,1.2] at all checked locations".
        # The checked locations are the validation a values.
        # However, for the integral, we use the fine grid. If gamma goes out of bounds there,
        # it might cause issues. We should probably clip gamma in the calculation for stability,
        # or ensure the ansatz keeps it in bounds.
        # Let's clip for safety in the integral calculation.
        gamma_fine = np.clip(gamma_fine, 0.1, 1.2)
        
        f_approx_fine = Om_fine ** gamma_fine
        
        # Integrate f_approx_fine dx from x_i to 0
        # Cumulative integral from x_fine[0] to x
        # We want integral from x to 0.
        # Total integral from x_min to 0 is I_total.
        # Integral from x to 0 is I_total - Integral from x_min to x.
        
        # Use trapezoidal rule
        # np.trapezoid is available in newer numpy, but let's use sum to be safe with versions
        # or check for trapezoid
        if hasattr(np, 'trapezoid'):
            cum_int = np.zeros_like(x_fine)
            # cum_int[i] = integral from x_fine[0] to x_fine[i]
            # We can do this iteratively or use cumulative sum of trapezoids
            # Trapezoid area between i-1 and i: 0.5 * (y[i-1] + y[i]) * dx
            # This is O(N).
            
            # Simpler: np.cumsum of trapezoids
            # d_int = 0.5 * (f[1:] + f[:-1]) * dx
            # cum_int[0] = 0
            # cum_int[1:] = cumsum(d_int)
            
            d_int = 0.5 * (f_approx_fine[1:] + f_approx_fine[:-1]) * dx_fine
            cum_int = np.concatenate(([0.0], np.cumsum(d_int)))
            
            # For each a_val (x_val), find the index in x_fine
            # x_fine is sorted.
            # We need integral from x_val to 0.
            # This is cum_int[-1] - cum_int[idx]
            
            # Interpolate cum_int to x_vals
            # np.interp works on y values.
            # We want I(x) = cum_int corresponding to x.
            # Since x_fine is uniform, we can just interpolate.
            cum_int_at_x_vals = np.interp(_REF_CACHE['x_vals'], x_fine, cum_int)
            
            I_total = cum_int[-1]
            integrals = I_total - cum_int_at_x_vals
            
            D_approx = np.exp(-integrals)
            
        else:
            # Fallback to manual trapezoid if np.trapezoid is missing (older numpy)
            d_int = 0.5 * (f_approx_fine[1:] + f_approx_fine[:-1]) * dx_fine
            cum_int = np.concatenate(([0.0], np.cumsum(d_int)))
            cum_int_at_x_vals = np.interp(_REF_CACHE['x_vals'], x_fine, cum_int)
            I_total = cum_int[-1]
            integrals = I_total - cum_int_at_x_vals
            D_approx = np.exp(-integrals)

        # Relative error in D
        rel_err_D = np.abs(D_approx - D_ref) / np.maximum(np.abs(D_ref), 1e-10)
        max_rel_err = max(max_rel_err, np.max(rel_err_D))
        
    return max_rel_err * 100

def construct():
    # Initialize cache
    _init_cache()
    
    start_time = time.time()
    deadline = start_time + 80  # 80s deadline, 10s safety margin
    
    # Initial guess: Linder approximation gamma = 0.55 + 0.1*q
    # c0=0.55, c1=0.1, others 0
    x0 = [0.55, 0.1, 0.0, 0.0, 0.0, 0.0]
    
    best_x = x0
    best_err = _growth_index_error(x0)
    
    # Optimize
    # We can try a few methods or just Nelder-Mead which is robust for small dims
    for i in range(10):
        if time.time() > deadline:
            break
        
        try:
            result = minimize(_growth_index_error, best_x, method='Nelder-Mead',
                             options={'maxiter': 200, 'xatol': 1e-6, 'fatol': 1e-6})
            if result.fun < best_err:
                best_err = result.fun
                best_x = result.x
        except Exception:
            break
    
    best_x = np.clip(best_x, -10, 10)
    
    return list(best_x)
# EVOLVE-BLOCK-END
