"""Growth-index approximation for CPL dark-energy cosmologies."""
import numpy as np
from scipy.optimize import minimize
from scipy.integrate import solve_ivp
import time

# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import minimize, differential_evolution
from scipy.integrate import solve_ivp

# Pre-computed reference data cache
_ref_data_cache = None

def _compute_om_a_vec(a, Om0, w0, wa):
    """Vectorized computation of matter density fraction."""
    a = np.asarray(a, dtype=np.float64)
    a_safe = np.where(a > 0, a, 1e-10)
    
    # Exact CPL evolution for DE density
    w_eff = 1 + w0 + wa
    de_term = (1 - Om0) * (a_safe ** (-3 * w_eff)) * np.exp(3 * wa * a_safe)
    m_term = Om0 * (a_safe ** -3)
    E2 = m_term + de_term
    Om_a = m_term / E2
    return Om_a

def _compute_gamma_vec(a, Om0, w0, wa, coeffs):
    """Vectorized computation of gamma."""
    c0, c1, c2, c3, c4, c5 = coeffs
    Om_a = _compute_om_a_vec(a, Om0, w0, wa)
    q = 1.0 - Om_a
    u = 1.0 + w0 + wa * (1.0 - a)
    gamma = c0 + c1 * q + c2 * q**2 + c3 * u + c4 * wa * a + c5 * u * q
    return gamma

def _compute_f_vec(a, Om0, w0, wa, coeffs):
    """Vectorized computation of f = Om(a)^gamma."""
    Om_a = _compute_om_a_vec(a, Om0, w0, wa)
    gamma = _compute_gamma_vec(a, Om0, w0, wa, coeffs)
    Om_a_safe = np.maximum(Om_a, 1e-10)
    return Om_a_safe ** gamma

def _compute_D_vec(a, Om0, w0, wa, coeffs):
    """
    Vectorized computation of normalized D.
    D(a)/D(1) = exp(-int_ln(a)^0 f(exp(x)) dx) = exp(int_0^ln(a) f(exp(x)) dx)
    """
    a = np.asarray(a, dtype=np.float64)
    D = np.ones_like(a)
    mask = a < 1.0
    if not np.any(mask):
        return D
    
    a_masked = a[mask]
    a_min = np.min(a_masked)
    
    # Grid for integration
    n_points = 400
    log_a_grid = np.linspace(np.log(a_min), 0.0, n_points)
    a_grid = np.exp(log_a_grid)
    
    f_grid = _compute_f_vec(a_grid, Om0, w0, wa, coeffs)
    
    # Cumulative integral from 0 to x (where x is negative)
    # Grid is ascending: log(a_min) ... 0
    # We want integral from 0 to x. 
    # Let's compute cumulative integral from the end (0) backwards.
    # Or simply: I(x) = - int_x^0 f(t) dt.
    # Let's compute cumulative trapezoid on the ascending grid.
    # C[i] = int_{log_a_grid[0]}^{log_a_grid[i]} f(t) dt
    # Then int_0^x = C[k] - C[-1] where x = log_a_grid[k].
    
    dx = np.diff(log_a_grid)
    # trapezoidal rule
    trapezoids = 0.5 * (f_grid[1:] + f_grid[:-1]) * dx
    cumulative = np.concatenate([[0.0], np.cumsum(trapezoids)])
    
    # cumulative[i] is integral from log_a_grid[0] to log_a_grid[i]
    # Total integral from log_a_grid[0] to 0 is cumulative[-1]
    # We want integral from 0 to x (x < 0).
    # int_0^x f(t) dt = - int_x^0 f(t) dt = - (int_{log_min}^0 - int_{log_min}^x) = int_{log_min}^x - int_{log_min}^0
    # int_{log_min}^x = cumulative[k]
    # int_{log_min}^0 = cumulative[-1]
    # So I(x) = cumulative[k] - cumulative[-1]
    
    # Interpolate cumulative integral values
    # xp is log_a_grid (ascending)
    # fp is cumulative
    ln_a_masked = np.log(a_masked)
    I_masked = np.interp(ln_a_masked, log_a_grid, cumulative) - cumulative[-1]
    
    D[mask] = np.exp(I_masked)
    return D

def _precompute_references():
    """Pre-compute reference f and D for a set of validation models."""
    global _ref_data_cache
    if _ref_data_cache is not None:
        return _ref_data_cache
        
    # Validation models
    models = [
        (0.3, -1.0, 0.0),   # LCDM
        (0.3, -0.9, 0.2),
        (0.3, -1.1, -0.2),
        (0.25, -1.0, 0.5),
        (0.35, -0.8, 0.3),
        (0.3, -1.5, 1.0),
        (0.3, -0.5, -0.5),
        (0.2, -1.2, 0.8),
        (0.4, -0.7, -0.3),
    ]
    
    a_vals = np.array([0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 1.0])
    
    ref_data = []
    for Om0, w0, wa in models:
        def ode_func(ln_a, y):
            f = y[0]
            a = np.exp(ln_a)
            Om_a = _compute_om_a_vec(a, Om0, w0, wa)[0]
            E2 = Om_a * a**(-3) + (1 - Om_a) * a**(-3*(1+w0+wa)) * np.exp(3*wa*a)
            
            # dE2/da
            g = a**(-3*(1+w0+wa)) * np.exp(3*wa*a)
            dE2_da = -3*Om0*a**(-4) + (1-Om0)*g*3*(wa - (1+w0+wa)/a)
            dlnE2_dlna = (dE2_da * a) / E2
            Hprime_H = 0.5 * dlnE2_dlna
            
            f_prime = -(1.5) * Om_a / E2 - (3.0 / a + Hprime_H) * f - f**2
            return [f_prime]
        
        f_init = 0.55 * Om0**0.55
        ln_a_grid = np.log(a_vals)
        
        try:
            sol = solve_ivp(ode_func, [0, np.log(0.2)], [f_init], t_eval=ln_a_grid, method='RK45', rtol=1e-9, atol=1e-11)
            if sol.success:
                f_ref = sol.y[0]
            else:
                f_ref = _compute_f_vec(a_vals, Om0, w0, wa, [0.55, 0, 0, 0, 0, 0])
        except Exception:
            f_ref = _compute_f_vec(a_vals, Om0, w0, wa, [0.55, 0, 0, 0, 0, 0])
            
        # Reference D
        ln_a_rev = ln_a_grid[::-1]
        f_rev = f_ref[::-1]
        
        D_ref = np.ones_like(a_vals)
        integral_vals = np.zeros_like(ln_a_rev)
        for i in range(1, len(ln_a_rev)):
            integral_vals[i] = integral_vals[i-1] + 0.5 * (f_rev[i] + f_rev[i-1]) * (ln_a_rev[i] - ln_a_rev[i-1])
        
        D_rev = np.exp(integral_vals)
        D_ref = D_rev[::-1]
        
        ref_data.append((Om0, w0, wa, a_vals, f_ref, D_ref))
        
    _ref_data_cache = ref_data
    return ref_data

def _evaluate_score(coeffs):
    """Calculate the worst-case relative error for a set of coefficients."""
    ref_data = _precompute_references()
    max_err = 0.0
    
    for Om0, w0, wa, a_vals, f_ref, D_ref in ref_data:
        f_approx = _compute_f_vec(a_vals, Om0, w0, wa, coeffs)
        D_approx = _compute_D_vec(a_vals, Om0, w0, wa, coeffs)
        
        err_f = np.abs(f_approx - f_ref) / np.maximum(np.abs(f_ref), 1e-10)
        max_err = max(max_err, np.max(err_f))
        
        err_D = np.abs(D_approx - D_ref) / np.maximum(np.abs(D_ref), 1e-10)
        max_err = max(max_err, np.max(err_D))
        
        gamma_vals = _compute_gamma_vec(a_vals, Om0, w0, wa, coeffs)
        if np.any(gamma_vals < 0.1) or np.any(gamma_vals > 1.2):
            return 1.0
    
    return max_err

def _optimize_coeffs():
    start_time = time.time()
    deadline = start_time + 75.0 # 15s margin
    
    # Initial guess
    best_coeffs = np.array([0.55, 0.0, 0.05, -0.05, 0.0, 0.03])
    best_score = _evaluate_score(best_coeffs)
    
    def objective(x):
        return _evaluate_score(x)
    
    bounds = [(-10, 10)] * 6
    
    # Run Nelder-Mead
    try:
        result = minimize(objective, best_coeffs, method='Nelder-Mead', 
                          options={'maxiter': 1000, 'xatol': 1e-6, 'fatol': 1e-7, 'adaptive': True})
        
        if result.fun < best_score:
            best_score = result.fun
            best_coeffs = result.x
            
    except Exception:
        pass
    
    if time.time() > deadline:
        return best_coeffs
    
    # Try Differential Evolution for global search if time permits
    if time.time() < deadline - 30:
        try:
            def de_objective(x):
                return _evaluate_score(x)
            
            # Use a subset of bounds for speed, or full bounds
            # DE can be slow, so limit iterations
            res = differential_evolution(de_objective, bounds, seed=42, maxiter=50, tol=1e-5, polish=True)
            
            if res.fun < best_score:
                best_score = res.fun
                best_coeffs = res.x
                
        except Exception:
            pass
    
    # Final polish with Nelder-Mead from best point
    if time.time() < deadline - 5:
        try:
            result = minimize(objective, best_coeffs, method='Nelder-Mead', 
                              options={'maxiter': 500, 'xatol': 1e-7, 'fatol': 1e-8, 'adaptive': True})
            if result.fun < best_score:
                best_score = result.fun
                best_coeffs = result.x
        except Exception:
            pass
    
    return best_coeffs

# Cache the optimized coefficients
_optimized_coeffs = None

def construct():
    global _optimized_coeffs
    if _optimized_coeffs is None:
        _optimized_coeffs = _optimize_coeffs()
    return _optimized_coeffs
# EVOLVE-BLOCK-END
