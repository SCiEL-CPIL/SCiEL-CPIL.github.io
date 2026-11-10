"""Growth-index formula optimizer for CPL dark-energy cosmologies."""
import time
import math
import numpy as np
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d

# Deadline guard
_START = time.time()
_DEADLINE = 85.0  # safety margin before 90s kill

def _elapsed_ok():
    return (time.time() - _START) < _DEADLINE

def _Om_of_a(a, Om0, w0, wa):
    """Matter density parameter as function of scale factor."""
    a = np.asarray(a, dtype=float)
    a_safe = np.maximum(a, 1e-10)
    de_exp = 3.0 * wa * (1.0/a_safe - 1.0)
    E2 = Om0 * a_safe**3 + (1.0 - Om0) * np.exp(de_exp)
    return Om0 * a_safe**3 / E2

def _gamma_from_coeffs(coeffs, a, Om0, w0, wa):
    """Compute gamma = c0 + c1*q + c2*q^2 + c3*u + c4*wa*a + c5*u*q"""
    c0, c1, c2, c3, c4, c5 = coeffs
    Om = _Om_of_a(a, Om0, w0, wa)
    q = 1.0 - Om
    u = 1.0 + w0 + wa * (1.0 - a)
    return c0 + c1*q + c2*q**2 + c3*u + c4*wa*a + c5*u*q

def _f_approx(coeffs, a, Om0, w0, wa):
    """f = Om(a)^gamma"""
    Om = _Om_of_a(a, Om0, w0, wa)
    g = _gamma_from_coeffs(coeffs, a, Om0, w0, wa)
    # Clamp gamma to [0.1, 1.2] for evaluation
    g = np.clip(g, 0.1, 1.2)
    return Om**g

def _D_normalized(coeffs, a, Om0, w0, wa):
    """D(a)/D(1) = exp(-integral from ln(a) to 0 of f(exp(x)) dx)"""
    # integral from ln(a) to 0 of f(exp(x)) dx = integral from a to 1 of f(t)/t dt
    if a >= 1.0:
        return 1.0
    # Integrate f(t)/t from a to 1
    n_pts = 200
    t = np.linspace(a, 1.0, n_pts)
    f_vals = _f_approx(coeffs, t, Om0, w0, wa)
    integrand = f_vals / t
    # Simpson's rule
    if n_pts % 2 == 1:
        h = (1.0 - a) / (n_pts - 1)
        integral = h/3.0 * (integrand[0] + integrand[-1] + 
                           4.0*np.sum(integrand[1:-1:2]) + 
                           2.0*np.sum(integrand[2:-1:2]))
    else:
        integral = np.trapz(integrand, t)
    return math.exp(-integral)

# Cache for ODE solutions
_ode_cache = {}

def _solve_ode_cached(Om0, w0, wa, a_grid=None):
    """Solve the GR ODE for f(a) with caching."""
    key = (round(Om0, 6), round(w0, 6), round(wa, 6))
    if key in _ode_cache:
        return _ode_cache[key]
    
    if not _elapsed_ok():
        return None
    
    if a_grid is None:
        a_grid = np.linspace(0.01, 1.0, 1000)
    
    def ode_rhs(a_val, f_val):
        Om = _Om_of_a(np.array([a_val]), Om0, w0, wa)[0]
        w = w0 + wa * (1.0 - a_val)
        # df/d ln a = -f^2 - (3/2)(1+w)f + (3/2) Om
        return [-f_val**2 - 1.5*(1.0+w)*f_val + 1.5*Om]
    
    # Initial condition: f ~ Om^0.55 in matter-dominated era
    a_start = 0.01
    Om_start = _Om_of_a(np.array([a_start]), Om0, w0, wa)[0]
    f_start = Om_start**0.55
    
    try:
        sol = solve_ivp(ode_rhs, [a_start, 1.0], [f_start], 
                       t_eval=a_grid, method='RK45', rtol=1e-10, atol=1e-12)
        if sol.success:
            interp_f = interp1d(a_grid, sol.y[0], kind='cubic', fill_value='extrapolate')
            _ode_cache[key] = interp_f
            return interp_f
        else:
            return None
    except Exception:
        return None

def _f_reference_ode(a, Om0, w0, wa):
    """Solve the GR ODE for f(a) = d ln D / d ln a."""
    interp_f = _solve_ode_cached(Om0, w0, wa)
    if interp_f is None:
        # Fallback: use approximate formula
        Om = _Om_of_a(np.array([a]), Om0, w0, wa)[0]
        return Om**0.55
    return float(interp_f(a))

def _D_reference_ode(a, Om0, w0, wa):
    """Compute reference D(a)/D(1) using ODE-solved f."""
    interp_f = _solve_ode_cached(Om0, w0, wa)
    if interp_f is None:
        return 1.0
    
    if a >= 1.0:
        return 1.0
    
    # Integrate f(t)/t from a to 1
    n_pts = 200
    t = np.linspace(a, 1.0, n_pts)
    f_vals = interp_f(t)
    integrand = f_vals / t
    integral = np.trapz(integrand, t)
    return math.exp(-integral)

def _objective(coeffs, cosmologies, a_values):
    """Compute max relative error over all cosmologies and a values."""
    max_err = 0.0
    for Om0, w0, wa in cosmologies:
        # Pre-solve ODE for this cosmology
        interp_f = _solve_ode_cached(Om0, w0, wa)
        
        for a in a_values:
            if a <= 0 or a >= 1.0:
                continue
            # Check gamma constraint
            g = _gamma_from_coeffs(coeffs, np.array([a]), Om0, w0, wa)[0]
            if g < 0.1 or g > 1.2:
                return 1e10  # penalty
            
            f_approx = _f_approx(coeffs, np.array([a]), Om0, w0, wa)[0]
            
            if interp_f is not None:
                f_ref = float(interp_f(a))
            else:
                f_ref = _f_reference_ode(a, Om0, w0, wa)
            
            if abs(f_ref) > 1e-8:
                err_f = abs(f_approx - f_ref) / abs(f_ref)
            else:
                err_f = abs(f_approx - f_ref)
            
            max_err = max(max_err, err_f)
            
            # D normalized error
            D_approx = _D_normalized(coeffs, np.array([a]), Om0, w0, wa)[0]
            D_ref = _D_reference_ode(a, Om0, w0, wa)
            
            if abs(D_ref) > 1e-8:
                err_D = abs(D_approx - D_ref) / abs(D_ref)
            else:
                err_D = abs(D_approx - D_ref)
            
            max_err = max(max_err, err_D)
    
    return max_err

def _optimize(coeffs_init, cosmologies, a_values, max_time=70.0):
    """Nelder-Mead style optimization with time guard."""
    start_time = time.time()
    best_coeffs = list(coeffs_init)
    best_err = _objective(best_coeffs, cosmologies, a_values)
    
    n_dim = len(coeffs_init)
    step_sizes = [0.03] * n_dim
    
    # Nelder-Mead simplex
    simplex = [list(coeffs_init)]
    for i in range(n_dim):
        pt = list(coeffs_init)
        pt[i] += step_sizes[i]
        simplex.append(pt)
    
    simplex_errs = [_objective(s, cosmologies, a_values) for s in simplex]
    
    max_iter = 300
    for iteration in range(max_iter):
        if time.time() - start_time > max_time:
            break
        if not _elapsed_ok():
            break
        
        # Sort simplex by error
        order = np.argsort(simplex_errs)
        simplex = [simplex[i] for i in order]
        simplex_errs = [simplex_errs[i] for i in order]
        
        # Check convergence
        if simplex_errs[0] < 1e-9:
            break
        
        # Centroid of all but worst
        centroid = np.mean(simplex[:-1], axis=0)
        worst = simplex[-1]
        worst_err = simplex_errs[-1]
        
        # Reflection
        alpha = 1.0
        x_r = centroid + alpha * (centroid - np.array(worst))
        x_r = list(x_r)
        err_r = _objective(x_r, cosmologies, a_values)
        
        if err_r < simplex_errs[0]:
            # Expansion
            gamma_exp = 2.0
            x_e = centroid + gamma_exp * (np.array(x_r) - centroid)
            x_e = list(x_e)
            err_e = _objective(x_e, cosmologies, a_values)
            if err_e < err_r:
                simplex[-1] = x_e
                simplex_errs[-1] = err_e
            else:
                simplex[-1] = x_r
                simplex_errs[-1] = err_r
        elif err_r < simplex_errs[-2]:
            simplex[-1] = x_r
            simplex_errs[-1] = err_r
        else:
            # Contraction
            rho = 0.5
            x_c = centroid + rho * (np.array(worst) - centroid)
            x_c = list(x_c)
            err_c = _objective(x_c, cosmologies, a_values)
            if err_c < worst_err:
                simplex[-1] = x_c
                simplex_errs[-1] = err_c
            else:
                # Shrink
                for i in range(1, n_dim + 1):
                    simplex[i] = [simplex[0][j] + 0.5*(simplex[i][j] - simplex[0][j]) for j in range(n_dim)]
                    simplex_errs[i] = _objective(simplex[i], cosmologies, a_values)
        
        # Update best
        min_idx = np.argmin(simplex_errs)
        if simplex_errs[min_idx] < best_err:
            best_err = simplex_errs[min_idx]
            best_coeffs = list(simplex[min_idx])
    
    # Polish with finer steps
    if _elapsed_ok() and (time.time() - start_time) < max_time - 5:
        for i in range(30):
            if not _elapsed_ok():
                break
            improved = False
            for j in range(n_dim):
                for delta in [0.002, -0.002, 0.0005, -0.0005]:
                    trial = list(best_coeffs)
                    trial[j] += delta
                    err = _objective(trial, cosmologies, a_values)
                    if err < best_err:
                        best_err = err
                        best_coeffs = trial
                        improved = True
            if not improved:
                break
    
    return best_coeffs, best_err

def construct():
    """Return 6 coefficients for the growth-index formula."""
    # Validation cosmologies (typical CPL models)
    cosmologies = [
        (0.3, -1.0, 0.0),    # Lambda CDM
        (0.3, -1.0, 0.3),    # CPL with wa=0.3
        (0.3, -1.0, -0.5),   # CPL with wa=-0.5
        (0.27, -1.0, 0.0),   # Lower Om0
        (0.3, -0.9, 0.0),    # w0 != -1
        (0.3, -1.1, 0.2),    # w0 < -1
        (0.35, -1.0, 0.0),   # Higher Om0
        (0.3, -1.0, 0.5),    # Strong wa
    ]
    
    # a values for evaluation
    a_values = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99]
    
    # Start from constant gamma
    coeffs_init = [0.55, 0.0, 0.0, 0.0, 0.0, 0.0]
    
    try:
        best_coeffs, best_err = _optimize(coeffs_init, cosmologies, a_values, max_time=70.0)
        
        # Ensure coefficients are in [-10, 10] and finite
        for i in range(6):
            if not math.isfinite(best_coeffs[i]):
                best_coeffs[i] = 0.0
            best_coeffs[i] = max(-10.0, min(10.0, best_coeffs[i]))
        
        return best_coeffs
    except Exception:
        return [0.55, 0.0, 0.0, 0.0, 0.0, 0.0]
