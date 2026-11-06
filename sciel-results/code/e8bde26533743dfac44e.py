"""Growth-index approximation for CPL dark-energy cosmologies."""
import numpy as np
from scipy.optimize import minimize
from scipy.integrate import solve_ivp
import time

# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import minimize, linprog
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
    
    # Grid for integration - use higher resolution for better accuracy
    n_points = 800
    log_a_grid = np.linspace(np.log(a_min), 0.0, n_points)
    a_grid = np.exp(log_a_grid)
    
    f_grid = _compute_f_vec(a_grid, Om0, w0, wa, coeffs)
    
    dx = np.diff(log_a_grid)
    # trapezoidal rule
    trapezoids = 0.5 * (f_grid[1:] + f_grid[:-1]) * dx
    cumulative = np.concatenate([[0.0], np.cumsum(trapezoids)])
    
    # Interpolate cumulative integral values
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

def _evaluate_score_smooth(coeffs, beta=50.0):
    """
    Smooth surrogate using log-sum-exp for the max over all errors.
    This is differentiable and allows better optimization.
    """
    ref_data = _precompute_references()
    all_errors = []
    
    for Om0, w0, wa, a_vals, f_ref, D_ref in ref_data:
        f_approx = _compute_f_vec(a_vals, Om0, w0, wa, coeffs)
        D_approx = _compute_D_vec(a_vals, Om0, w0, wa, coeffs)
        
        err_f = np.abs(f_approx - f_ref) / np.maximum(np.abs(f_ref), 1e-10)
        err_D = np.abs(D_approx - D_ref) / np.maximum(np.abs(D_ref), 1e-10)
        
        all_errors.append(err_f)
        all_errors.append(err_D)
        
        gamma_vals = _compute_gamma_vec(a_vals, Om0, w0, wa, coeffs)
        if np.any(gamma_vals < 0.1) or np.any(gamma_vals > 1.2):
            return 1.0
    
    all_errors = np.concatenate(all_errors)
    
    # Log-sum-exp for smooth max
    m = np.max(all_errors)
    if m < 1e-15:
        return 0.0
    smooth_max = m + np.log(np.sum(np.exp(all_errors - m))) / beta
    return smooth_max

def _optimize_coeffs_lp():
    """
    Solve for coefficients using Linear Programming to minimize max log-error of f.
    Variables: c0..c5, epsilon
    Minimize epsilon
    Subject to:
    -epsilon <= ln(f_approx) - ln(f_ref) <= epsilon
    0.1 <= gamma <= 1.2
    """
    ref_data = _precompute_references()
    
    # Collect all points
    points = []
    for Om0, w0, wa, a_vals, f_ref, D_ref in ref_data:
        for i in range(len(a_vals)):
            a = a_vals[i]
            points.append((a, Om0, w0, wa, f_ref[i]))
    
    n_points = len(points)
    # Variables: c0, c1, c2, c3, c4, c5, epsilon (index 6)
    
    # Objective: minimize epsilon
    c_obj = np.zeros(7)
    c_obj[6] = 1.0
    
    # Constraints A_ub x <= b_ub
    A_rows = []
    b_rows = []
    
    for a, Om0, w0, wa, f_r in points:
        Om_a = _compute_om_a_vec(a, Om0, w0, wa)
        q = 1.0 - Om_a
        u = 1.0 + w0 + wa * (1.0 - a)
        ln_Om = np.log(np.maximum(Om_a, 1e-10))
        ln_f_ref = np.log(np.maximum(f_r, 1e-10))
        
        # Coefficients for gamma
        basis = np.array([1.0, q, q**2, u, wa * a, u * q])
        
        # Constraint 1: gamma * ln_Om - ln_f_ref <= epsilon
        # c0*ln_Om + ... + c5*... - epsilon <= ln_f_ref
        row1 = np.zeros(7)
        row1[:6] = basis * ln_Om
        row1[6] = -1.0
        A_rows.append(row1)
        b_rows.append(ln_f_ref)
        
        # Constraint 2: -(gamma * ln_Om - ln_f_ref) <= epsilon
        # -c0*ln_Om - ... - c5*... - epsilon <= -ln_f_ref
        row2 = np.zeros(7)
        row2[:6] = -basis * ln_Om
        row2[6] = -1.0
        A_rows.append(row2)
        b_rows.append(-ln_f_ref)
        
        # Constraint 3: gamma >= 0.1  => -gamma <= -0.1
        row3 = np.zeros(7)
        row3[:6] = -basis
        A_rows.append(row3)
        b_rows.append(-0.1)
        
        # Constraint 4: gamma <= 1.2  => gamma <= 1.2
        row4 = np.zeros(7)
        row4[:6] = basis
        A_rows.append(row4)
        b_rows.append(1.2)
        
    A_ub = np.array(A_rows)
    b_ub = np.array(b_rows)
    
    # Bounds for c0..c5 in [-10, 10], epsilon >= 0
    bounds = [(-10, 10)] * 6 + [(0, None)]
    
    try:
        res = linprog(c_obj, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
        if res.success:
            return res.x[:6]
    except Exception:
        pass
    
    return np.array([0.55, 0.0, 0.05, -0.05, 0.0, 0.03])

def _optimize_coeffs():
    start_time = time.time()
    deadline = start_time + 85.0  # 5s margin for safety
    
    # Phase 1: LP to get a good initial guess for f-error
    best_coeffs = _optimize_coeffs_lp()
    best_score = _evaluate_score(best_coeffs)
    
    # Phase 2: Use smooth surrogate with Nelder-Mead for better convergence
    def objective_smooth(x):
        return _evaluate_score_smooth(x, beta=50.0)
    
    try:
        result = minimize(objective_smooth, best_coeffs, method='Nelder-Mead', 
                          options={'maxiter': 3000, 'xatol': 1e-7, 'fatol': 1e-8, 'adaptive': True})
        
        if result.fun < best_score:
            # Verify with exact score
            exact_score = _evaluate_score(result.x)
            if exact_score < best_score:
                best_score = exact_score
                best_coeffs = result.x.copy()
            elif exact_score < 1.0:
                best_coeffs = result.x.copy()
                
    except Exception:
        pass
    
    if time.time() > deadline:
        return best_coeffs
    
    # Phase 3: Refine with exact objective using tighter tolerances
    if time.time() < deadline - 5:
        def objective_exact(x):
            return _evaluate_score(x)
        
        try:
            result = minimize(objective_exact, best_coeffs, method='Nelder-Mead', 
                              options={'maxiter': 2000, 'xatol': 1e-8, 'fatol': 1e-9, 'adaptive': True})
            
            if result.fun < best_score:
                best_score = result.fun
                best_coeffs = result.x.copy()
                
        except Exception:
            pass
    
    # Phase 4: Final polish with even tighter tolerance
    if time.time() < deadline - 2:
        def objective_exact(x):
            return _evaluate_score(x)
        
        try:
            result = minimize(objective_exact, best_coeffs, method='Nelder-Mead', 
                              options={'maxiter': 1000, 'xatol': 1e-9, 'fatol': 1e-10, 'adaptive': True})
            
            if result.fun < best_score:
                best_score = result.fun
                best_coeffs = result.x.copy()
                
        except Exception:
            pass
    
    return best_coeffs

def construct():
    """
    Construct the six coefficients for the growth-index approximation.
    Returns a list of six finite coefficients in [-10, 10].
    """
    start_time = time.time()
    
    try:
        coeffs = _optimize_coeffs()
        coeffs = np.clip(coeffs, -10, 10)
        return list(coeffs)
    except Exception:
        # Fallback to seed values
        return [0.55, 0.0, 0.0, 0.0, 0.0, 0.0]
# EVOLVE-BLOCK-END
