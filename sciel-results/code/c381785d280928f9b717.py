"""Growth-index formula optimizer for CPL dark-energy cosmologies."""
import time
import math
import numpy as np
from scipy.optimize import minimize

_START = time.time()
_DEADLINE = 85.0

def _elapsed_ok():
    return (time.time() - _START) < _DEADLINE

def _Om_of_a(a, Om0, w0, wa):
    a = np.asarray(a, dtype=float)
    a_safe = np.maximum(a, 1e-10)
    de_exp = 3.0 * wa * (1.0 / a_safe - 1.0)
    E2 = Om0 * a_safe**3 + (1.0 - Om0) * np.exp(de_exp)
    return Om0 * a_safe**3 / E2

def _solve_ode_f(a_grid, Om0, w0, wa):
    """Solve GR ODE for f(a) on a grid, return f values."""
    a_start = 0.01
    a_end = 1.0
    n_steps = 2000
    h = (a_end - a_start) / n_steps
    a_grid_full = np.linspace(a_start, a_end, n_steps + 1)
    
    def f_of_a(a_val):
        Om = _Om_of_a(np.array([a_val]), Om0, w0, wa)[0]
        return Om**0.55
    
    def ode_rhs(a_val, f_val):
        Om = _Om_of_a(np.array([a_val]), Om0, w0, wa)[0]
        w = w0 + wa * (1.0 - a_val)
        return -f_val**2 - 1.5*(1.0 + w)*f_val + 1.5*Om
    
    f_grid = np.zeros(n_steps + 1)
    f_grid[0] = f_of_a(a_start)
    
    for i in range(n_steps):
        a_curr = a_grid_full[i]
        f_curr = f_grid[i]
        def df_da(av, fv):
            return ode_rhs(av, fv) / av
        k1 = df_da(a_curr, f_curr)
        k2 = df_da(a_curr + h/2, f_curr + h/2*k1)
        k3 = df_da(a_curr + h/2, f_curr + h/2*k2)
        k4 = df_da(a_curr + h, f_curr + h*k3)
        f_grid[i+1] = f_curr + h/6.0*(k1 + 2*k2 + 2*k3 + k4)
    
    # Interpolate to requested a_grid
    result = np.empty_like(a_grid)
    for i, a in enumerate(a_grid):
        if a <= a_start:
            result[i] = f_grid[0]
        elif a >= a_end:
            result[i] = f_grid[-1]
        else:
            idx = int((a - a_start) / h)
            if idx >= n_steps:
                idx = n_steps - 1
            frac = (a - a_grid_full[idx]) / h
            result[i] = f_grid[idx] + frac * (f_grid[idx+1] - f_grid[idx])
    return result

def _compute_reference_D(a_grid, f_grid, Om0, w0, wa):
    """Compute normalized D(a) from f_grid by integrating f/t da."""
    n = len(a_grid)
    D = np.ones(n)
    if n < 2:
        return D
    # Integrate from a to 1: D(a)/D(1) = exp(-int_a^1 f(t)/t dt)
    # Precompute cumulative integral from right
    integrand = f_grid / a_grid
    # trapezoidal cumulative from right
    cum = np.zeros(n)
    cum[-1] = 0.0
    for i in range(n-2, -1, -1):
        h = a_grid[i+1] - a_grid[i]
        cum[i] = cum[i+1] + 0.5*h*(integrand[i] + integrand[i+1])
    D = np.exp(-cum)
    return D

def _build_reference_cache(cosmologies, a_values):
    """Pre-compute reference f and D for all cosmologies."""
    cache = {}
    for Om0, w0, wa in cosmologies:
        a_arr = np.array(a_values, dtype=float)
        f_ref = _solve_ode_f(a_arr, Om0, w0, wa)
        D_ref = _compute_reference_D(a_arr, f_ref, Om0, w0, wa)
        Om_arr = _Om_of_a(a_arr, Om0, w0, wa)
        q_arr = 1.0 - Om_arr
        u_arr = 1.0 + w0 + wa*(1.0 - a_arr)
        cache[(Om0, w0, wa)] = {
            'a': a_arr, 'f_ref': f_ref, 'D_ref': D_ref,
            'Om': Om_arr, 'q': q_arr, 'u': u_arr
        }
    return cache

def _gamma_from_coeffs(coeffs, q, u, wa, a):
    c0, c1, c2, c3, c4, c5 = coeffs
    return c0 + c1*q + c2*q**2 + c3*u + c4*wa*a + c5*u*q

def _objective_smooth(coeffs, cache, beta=30.0):
    """Smooth max relative error via log-sum-exp."""
    errors = []
    for key, ref in cache.items():
        a = ref['a']
        q = ref['q']
        u = ref['u']
        Om = ref['Om']
        f_ref = ref['f_ref']
        D_ref = ref['D_ref']
        Om0, w0, wa = key
        
        g = _gamma_from_coeffs(coeffs, q, u, wa, a)
        if np.any(g < 0.1) or np.any(g > 1.2):
            return 1e10
        
        f_approx = np.power(Om, g)
        
        # f errors
        mask_f = np.abs(f_ref) > 1e-8
        if np.any(mask_f):
            err_f = np.abs(f_approx[mask_f] - f_ref[mask_f]) / np.abs(f_ref[mask_f])
            errors.extend(err_f.tolist())
        
        # D errors: integrate f_approx/t from a to 1
        for i in range(len(a)):
            if a[i] >= 1.0:
                continue
            # Integrate f_approx(t)/t from a[i] to 1 using the grid
            # Use trapezoidal on the grid points from i to end
            f_approx_full = np.power(Om, g)
            integrand = f_approx_full / a
            # cumulative from i to end
            h_vals = np.diff(a)
            trap = 0.5 * (integrand[i:] + np.roll(integrand[i:], 1)) * h_vals
            # fix last element
            if len(trap) > 0:
                trap[-1] = 0.5 * integrand[-1] * h_vals[-1]
            integral_approx = np.sum(trap)
            D_approx = math.exp(-integral_approx)
            if abs(D_ref[i]) > 1e-8:
                err_D = abs(D_approx - D_ref[i]) / abs(D_ref[i])
                errors.append(err_D)
    
    if not errors:
        return 0.0
    errors_arr = np.array(errors)
    max_e = np.max(errors_arr)
    if max_e < 1e-15:
        return max_e
    # log-sum-exp
    shifted = errors_arr - max_e
    smooth_max = max_e + np.log(np.sum(np.exp(beta * shifted))) / beta
    return smooth_max

def _objective_exact(coeffs, cache):
    """Exact max relative error."""
    max_err = 0.0
    for key, ref in cache.items():
        a = ref['a']
        q = ref['q']
        u = ref['u']
        Om = ref['Om']
        f_ref = ref['f_ref']
        D_ref = ref['D_ref']
        Om0, w0, wa = key
        
        g = _gamma_from_coeffs(coeffs, q, u, wa, a)
        if np.any(g < 0.1) or np.any(g > 1.2):
            return 1e10
        
        f_approx = np.power(Om, g)
        
        mask_f = np.abs(f_ref) > 1e-8
        if np.any(mask_f):
            err_f = np.abs(f_approx[mask_f] - f_ref[mask_f]) / np.abs(f_ref[mask_f])
            max_err = max(max_err, np.max(err_f))
        
        f_approx_full = f_approx
        integrand = f_approx_full / a
        for i in range(len(a)):
            if a[i] >= 1.0:
                continue
            h_vals = np.diff(a)
            trap = 0.5 * (integrand[i:] + np.roll(integrand[i:], 1)) * h_vals
            if len(trap) > 0:
                trap[-1] = 0.5 * integrand[-1] * h_vals[-1]
            integral_approx = np.sum(trap)
            D_approx = math.exp(-integral_approx)
            if abs(D_ref[i]) > 1e-8:
                err_D = abs(D_approx - D_ref[i]) / abs(D_ref[i])
                max_err = max(max_err, err_D)
    
    return max_err

def _taylor_init(cache):
    """Fit coefficients using Taylor expansion around central cosmology."""
    Om0_c, w0_c, wa_c = 0.3, -1.0, 0.0
    key = (Om0_c, w0_c, wa_c)
    if key not in cache:
        return [0.55, 0.0, 0.0, 0.0, 0.0, 0.0]
    
    ref = cache[key]
    a = ref['a']
    f_ref = ref['f_ref']
    Om = ref['Om']
    q = ref['q']
    u = ref['u']
    
    # gamma_ref = ln(f_ref) / ln(Om)
    gamma_ref = np.zeros_like(f_ref)
    for i in range(len(f_ref)):
        if Om[i] > 0 and f_ref[i] > 0 and Om[i] != 1.0:
            gamma_ref[i] = np.log(f_ref[i]) / np.log(Om[i])
        else:
            gamma_ref[i] = 0.55
    
    A = np.zeros((len(a), 6))
    A[:, 0] = 1.0
    A[:, 1] = q
    A[:, 2] = q**2
    A[:, 3] = u
    A[:, 4] = wa_c * a
    A[:, 5] = u * q
    
    coeffs, _, _, _ = np.linalg.lstsq(A, gamma_ref, rcond=None)
    coeffs = np.clip(coeffs, -10.0, 10.0)
    return list(coeffs)

def construct():
    cosmologies = [
        (0.3, -1.0, 0.0),
        (0.3, -1.0, 0.3),
        (0.3, -1.0, -0.5),
        (0.27, -1.0, 0.0),
        (0.3, -0.9, 0.0),
        (0.3, -1.1, 0.2),
        (0.35, -1.0, 0.0),
        (0.3, -1.0, 0.5),
    ]
    a_values = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99]
    
    # Build reference cache (expensive, do once)
    cache = _build_reference_cache(cosmologies, a_values)
    
    best_coeffs = [0.55, 0.0, 0.0, 0.0, 0.0, 0.0]
    best_err = _objective_exact(best_coeffs, cache)
    
    # Taylor init
    try:
        taylor = _taylor_init(cache)
        taylor_err = _objective_exact(taylor, cache)
        if taylor_err < best_err:
            best_coeffs = taylor
            best_err = taylor_err
    except Exception:
        pass
    
    # Multi-start Nelder-Mead with smooth objective
    start_points = [
        list(best_coeffs),
        [0.5, 0.1, 0.0, 0.0, 0.0, 0.0],
        [0.6, -0.1, 0.0, 0.0, 0.0, 0.0],
        [0.55, 0.0, 0.1, 0.0, 0.0, 0.0],
        [0.55, 0.05, -0.05, 0.0, 0.0, 0.0],
    ]
    
    for sp in start_points:
        if not _elapsed_ok():
            break
        try:
            result = minimize(
                lambda x: _objective_smooth(x, cache, beta=30.0),
                sp, method='Nelder-Mead',
                options={'maxiter': 300, 'xatol': 1e-6, 'fatol': 1e-8, 'adaptive': True}
            )
            c = result.x
            if all(-10 <= c[i] <= 10 for i in range(6)) and all(math.isfinite(c[i]) for i in range(6)):
                c = [max(-10.0, min(10.0, c[i])) for i in range(6)]
                err = _objective_exact(c, cache)
                if err < best_err:
                    best_err = err
                    best_coeffs = c
        except Exception:
            pass
    
    # Final polish with coordinate descent
    if _elapsed_ok():
        for _ in range(100):
            if not _elapsed_ok():
                break
            improved = False
            for j in range(6):
                for delta in [0.01, -0.01, 0.005, -0.005, 0.002, -0.002, 0.001, -0.001]:
                    trial = list(best_coeffs)
                    trial[j] += delta
                    if trial[j] < -10 or trial[j] > 10:
                        continue
                    err = _objective_exact(trial, cache)
                    if err < best_err:
                        best_err = err
                        best_coeffs = trial
                        improved = True
            if not improved:
                break
    
    # Ensure validity
    for i in range(6):
        if not math.isfinite(best_coeffs[i]):
            best_coeffs[i] = 0.0
        best_coeffs[i] = max(-10.0, min(10.0, best_coeffs[i]))
    
    return best_coeffs
