"""Cosmology: compact dark-energy growth approximation.
Improve a six-coefficient growth-index formula for the linear growth rate and normalized
growth factor in smooth CPL dark-energy cosmologies.
"""
# EVOLVE-BLOCK-START
import time
import math
import numpy as np
from scipy.integrate import solve_ivp

_DEADLINE = 80.0
_start_time = time.time()
_ode_cache = {}


def _elapsed():
    return time.time() - _start_time


def _clear_ode_cache():
    global _ode_cache
    _ode_cache = {}


def _gamma_from_coeffs(coeffs, a, Om, w0, wa):
    c0, c1, c2, c3, c4, c5 = coeffs
    q = 1.0 - Om
    u = 1.0 + w0 + wa * (1.0 - a)
    return c0 + c1 * q + c2 * q * q + c3 * u + c4 * wa * a + c5 * u * q


def _f_value(coeffs, a, Om, w0, wa):
    gamma = _gamma_from_coeffs(coeffs, a, Om, w0, wa)
    if Om <= 0:
        return 1.0
    return Om ** gamma


def _H2(a, Om1, w0, wa):
    if a <= 0:
        return 0.0
    term_matter = Om1 * a ** (-3)
    if (1 - Om1) < 0:
        term_de = 0.0
    else:
        exponent = -3 * (1 + w0) + 3 * wa * (1 - a)
        if exponent > 700:
            term_de = (1 - Om1) * 1e300
        elif exponent < -700:
            term_de = 0.0
        else:
            term_de = (1 - Om1) * a ** (-3 * (1 + w0)) * math.exp(exponent)
    return term_matter + term_de


def _Om_of_a(a, Om1, w0, wa):
    h2 = _H2(a, Om1, w0, wa)
    if h2 <= 0:
        return 0.0
    return Om1 * a ** (-3) / h2


def _dlnH2_dx(a, Om1, w0, wa):
    h2 = _H2(a, Om1, w0, wa)
    if h2 <= 0:
        return 0.0
    term1 = -3 * Om1 * a ** (-4)
    if (1 - Om1) == 0:
        term2 = 0.0
    else:
        exp_part = math.exp(3 * wa * (1 - a))
        pow_part = a ** (-3 * (1 + w0) - 1)
        term2 = (1 - Om1) * pow_part * exp_part * (-3) * (1 + w0 + wa)
    h2_prime = term1 + term2
    return a * h2_prime / h2


def _solve_D_ode(a_start, a_end, Om1, w0, wa):
    """
    Solve D'' + (3/2 + 1/2 * dlnH2/dx) D' - (3/2) Om(x) D = 0
    where x = ln a.
    Returns D at a_end given D(a_start)=1, D'(a_start)=f(a_start).
    """
    if a_start <= 0 or a_end <= 0:
        return 0.0, 0.0
    
    x_start = math.log(a_start)
    x_end = math.log(a_end)
    
    # Initial conditions
    Om_s = _Om_of_a(a_start, Om1, w0, wa)
    D0 = 1.0
    Dp0 = Om_s ** 0.55 if Om_s > 0 else 0.0
    
    def ode(x, y):
        D, Dp = y
        a = math.exp(x)
        Om_a = _Om_of_a(a, Om1, w0, wa)
        dlnH2 = _dlnH2_dx(a, Om1, w0, wa)
        Dpp = -(1.5 + 0.5 * dlnH2) * Dp + 1.5 * Om_a * D
        return [Dp, Dpp]
    
    try:
        sol = solve_ivp(ode, [x_start, x_end], [D0, Dp0], method='RK45',
                        max_step=0.01, rtol=1e-10, atol=1e-12)
        if sol.success:
            return sol.y[0, -1], sol.y[1, -1]
        else:
            return 0.0, 0.0
    except:
        return 0.0, 0.0


def _solve_D_ode_cached(a_start, a_end, Om1, w0, wa):
    """Cached version of _solve_D_ode."""
    key = (round(a_start, 6), round(a_end, 6), round(Om1, 6), round(w0, 6), round(wa, 6))
    if key in _ode_cache:
        return _ode_cache[key]
    result = _solve_D_ode(a_start, a_end, Om1, w0, wa)
    _ode_cache[key] = result
    return result


def _D_normalized_ref(a, Om1, w0, wa):
    """
    Compute D(a)/D(1) using ODE reference.
    """
    if a <= 0:
        return 0.0
    if abs(a - 1.0) < 1e-10:
        return 1.0
    if a > 1.0:
        D_end, _ = _solve_D_ode_cached(1.0, a, Om1, w0, wa)
        return D_end
    else:
        # Integrate from a to 1
        D_end, _ = _solve_D_ode_cached(a, 1.0, Om1, w0, wa)
        # D_end is D(1) given D(a)=1, so D(a)/D(1) = 1/D_end
        if D_end != 0:
            return 1.0 / D_end
        return 0.0


def _f_ref(a, Om1, w0, wa):
    """Reference f = D'/D computed from ODE."""
    if a <= 0:
        return 0.0
    if abs(a - 1.0) < 1e-10:
        Om_a = _Om_of_a(1.0, Om1, w0, wa)
        return Om_a ** 0.55 if Om_a > 0 else 0.0
    
    # Integrate from 1 to a (or a to 1) to get D(a) and D'(a)
    if a > 1.0:
        D_a, Dp_a = _solve_D_ode_cached(1.0, a, Om1, w0, wa)
    else:
        # Integrate from a to 1, then D(a)/D(1) = 1/D_end
        D_end, Dp_end = _solve_D_ode_cached(a, 1.0, Om1, w0, wa)
        if D_end == 0:
            return 0.0
        # D(a) = 1, D(1) = D_end
        # D'(a) = Dp_a, D'(1) = Dp_end
        # f(a) = D'(a)/D(a) = Dp_a / 1 = Dp_a
        # But we need Dp_a, not Dp_end
        # Let's integrate from 1 to a instead for consistency
        D_a, Dp_a = _solve_D_ode_cached(1.0, a, Om1, w0, wa)
    
    if D_a != 0:
        return Dp_a / D_a
    return 0.0


def _compute_max_error(coeffs, test_models, a_vals):
    """
    Compute max relative error of f and D over test models and a values.
    """
    max_rel_err = 0.0
    for Om1, w0, wa in test_models:
        for a in a_vals:
            Om_a = _Om_of_a(a, Om1, w0, wa)
            
            # f error
            f_pred = _f_value(coeffs, a, Om_a, w0, wa)
            f_true = _f_ref(a, Om1, w0, wa)
            if abs(f_true) > 1e-15:
                rel_err_f = abs(f_pred - f_true) / abs(f_true)
                max_rel_err = max(max_rel_err, rel_err_f)
            
            # D error
            D_pred = _D_normalized(coeffs, a, Om1, w0, wa)
            D_true = _D_normalized_ref(a, Om1, w0, wa)
            if abs(D_true) > 1e-15:
                rel_err_D = abs(D_pred - D_true) / abs(D_true)
                max_rel_err = max(max_rel_err, rel_err_D)
    return max_rel_err


def _soft_max_error(coeffs, test_models, a_vals, beta=50.0):
    """
    Compute soft-max (log-sum-exp) of relative errors for smooth optimization.
    Returns a smooth surrogate for the max error.
    """
    errors = []
    for Om1, w0, wa in test_models:
        for a in a_vals:
            Om_a = _Om_of_a(a, Om1, w0, wa)
            
            # f error
            f_pred = _f_value(coeffs, a, Om_a, w0, wa)
            f_true = _f_ref(a, Om1, w0, wa)
            if abs(f_true) > 1e-15:
                rel_err_f = abs(f_pred - f_true) / abs(f_true)
                errors.append(rel_err_f)
            
            # D error
            D_pred = _D_normalized(coeffs, a, Om1, w0, wa)
            D_true = _D_normalized_ref(a, Om1, w0, wa)
            if abs(D_true) > 1e-15:
                rel_err_D = abs(D_pred - D_true) / abs(D_true)
                errors.append(rel_err_D)
    
    if not errors:
        return 0.0
    
    errors = np.array(errors)
    # Log-sum-exp for numerical stability
    max_e = np.max(errors)
    if max_e < 1e-300:
        return 0.0
    log_sum = np.log(np.sum(np.exp(beta * (errors - max_e)))) / beta
    return max_e + log_sum


def _D_normalized(coeffs, a, Om1, w0, wa, n_pts=200):
    """
    Compute D(a)/D(1) = exp(-integral from ln(a) to ln(1) f(exp(x)) dx)
    = exp(-integral from ln(a) to 0 f(exp(x)) dx)
    """
    if a <= 0:
        return 0.0
    if abs(a - 1.0) < 1e-10:
        return 1.0
    
    if a > 1.0:
        # Integral from 0 to ln(a)
        x_vals = np.linspace(0, math.log(a), n_pts)
    else:
        # Integral from ln(a) to 0
        x_vals = np.linspace(math.log(a), 0.0, n_pts)
    
    fs = []
    for x in x_vals:
        a_x = math.exp(x)
        Om_x = _Om_of_a(a_x, Om1, w0, wa)
        f_x = _f_value(coeffs, a_x, Om_x, w0, wa)
        fs.append(f_x)
    
    fs = np.array(fs)
    integral = np.trapz(fs, x_vals)
    
    if a > 1.0:
        return math.exp(integral)
    else:
        return math.exp(-integral)


def _validate_coeffs(coeffs, a_test_points, Om1, w0, wa):
    """Check that gamma is in [0.1, 1.2] at all test points. Return True if valid."""
    if not all(np.isfinite(c) for c in coeffs):
        return False
    if not all(-10 <= c <= 10 for c in coeffs):
        return False
    
    for a in a_test_points:
        Om_a = _Om_of_a(a, Om1, w0, wa)
        gamma = _gamma_from_coeffs(coeffs, a, Om_a, w0, wa)
        if not (0.1 <= gamma <= 1.2):
            return False
    return True


def _validate_all_models(coeffs, test_models, a_test_points):
    """Check gamma range across all models."""
    for Om1, w0, wa in test_models:
        if not _validate_coeffs(coeffs, a_test_points, Om1, w0, wa):
            return False
    return True


def _analytical_initialization():
    """
    Derive initial coefficients by matching the exact growth index expansion
    near the matter-dominated limit (q -> 1, u -> 1).
    
    For a flat Lambda CDM (w0=-1, wa=0), the exact growth index can be
    approximated as:
      gamma(q) ≈ 0.55 + 0.3*(1-q) + 0.15*(1-q)^2 + ...
    
    We match the first few terms to our ansatz:
      gamma = c0 + c1*q + c2*q^2 + c3*u + c4*wa*a + c5*u*q
    
    At the matter-dominated limit (a->0, q->1, u->1+w0):
      For w0=-1, u->0, so c3*u and c5*u*q vanish.
      gamma ≈ c0 + c1*q + c2*q^2
    
    Matching gamma(q) = 0.55 + 0.3*(1-q) + 0.15*(1-q)^2:
      = 0.55 + 0.3 - 0.3*q + 0.15*(1 - 2*q + q^2)
      = 0.55 + 0.3 + 0.15 - 0.3*q - 0.3*q + 0.15*q^2
      = 1.0 - 0.6*q + 0.15*q^2
    
    So: c0=1.0, c1=-0.6, c2=0.15
    
    For the u and wa terms, we use small values based on perturbation theory.
    """
    # Base coefficients from matter-dominated expansion
    c0 = 1.0
    c1 = -0.6
    c2 = 0.15
    # Small corrections for u and wa dependence
    c3 = 0.05
    c4 = 0.02
    c5 = 0.03
    return [c0, c1, c2, c3, c4, c5]


def construct():
    global _start_time
    _start_time = time.time()
    _clear_ode_cache()
    
    # Test cosmologies - include a wider range
    test_models = [
        (0.3, -1.0, 0.0),
        (0.27, -1.0, 0.0),
        (0.3, -1.0, 0.5),
        (0.3, -1.0, -0.5),
        (0.2, -1.2, 0.3),
        (0.4, -0.9, 0.1),
        (0.3, -1.0, 1.0),
        (0.3, -1.0, -1.0),
        (0.25, -0.8, 0.2),
        (0.35, -1.1, -0.3),
    ]
    
    a_vals = np.linspace(0.1, 1.0, 25)
    
    # Start from analytical initialization
    best_coeffs = _analytical_initialization()
    best_err = float('inf')
    
    # Compute initial error
    try:
        best_err = _compute_max_error(best_coeffs, test_models, a_vals)
    except:
        best_err = 1.0
        best_coeffs = [0.55, 0.0, 0.0, 0.0, 0.0, 0.0]
    
    # Phase 1: Coarse search with soft-max surrogate (smooth objective)
    if _elapsed() < _DEADLINE - 30:
        coarse_stages = [
            (0.1, 0.05, 0.02),
            (0.05, 0.02, 0.01),
            (0.02, 0.01, 0.005),
        ]
        
        for stage_idx, step_sizes in enumerate(coarse_stages):
            if _elapsed() > _DEADLINE - 30:
                break
            
            stage_improved = False
            max_iter_per_stage = 500
            for _ in range(max_iter_per_stage):
                if _elapsed() > _DEADLINE - 30:
                    break
                improved = False
                for i in range(6):
                    for sign in [-1, 1]:
                        for step in step_sizes:
                            candidate = best_coeffs.copy()
                            candidate[i] += sign * step
                            if not all(-10 <= c <= 10 for c in candidate):
                                continue
                            
                            # Validate gamma range
                            if not _validate_all_models(candidate, test_models, a_vals):
                                continue
                            
                            # Use soft-max for smooth optimization
                            try:
                                err = _soft_max_error(candidate, test_models, a_vals, beta=50.0)
                            except:
                                continue
                            
                            if err < best_err - 1e-10:
                                best_err = err
                                best_coeffs = candidate
                                improved = True
                                stage_improved = True
                                break
                        if improved:
                            break
                    if improved:
                        break
                if not improved:
                    break
            
            if not stage_improved and stage_idx > 0:
                break
    
    # Phase 2: Fine search with exact max error (nonsmooth)
    if _elapsed() < _DEADLINE - 10:
        # Recompute best_err with exact max
        try:
            best_err = _compute_max_error(best_coeffs, test_models, a_vals)
        except:
            best_err = 1.0
        
        fine_stages = [
            (0.01, 0.005, 0.002),
            (0.005, 0.002, 0.001),
            (0.002, 0.001, 0.0005),
        ]
        
        for stage_idx, step_sizes in enumerate(fine_stages):
            if _elapsed() > _DEADLINE - 10:
                break
            
            stage_improved = False
            max_iter_per_stage = 800
            for _ in range(max_iter_per_stage):
                if _elapsed() > _DEADLINE - 10:
                    break
                improved = False
                for i in range(6):
                    for sign in [-1, 1]:
                        for step in step_sizes:
                            candidate = best_coeffs.copy()
                            candidate[i] += sign * step
                            if not all(-10 <= c <= 10 for c in candidate):
                                continue
                            
                            # Validate gamma range
                            if not _validate_all_models(candidate, test_models, a_vals):
                                continue
                            
                            # Use exact max error
                            try:
                                err = _compute_max_error(candidate, test_models, a_vals)
                            except:
                                continue
                            
                            if err < best_err - 1e-12:
                                best_err = err
                                best_coeffs = candidate
                                improved = True
                                stage_improved = True
                                break
                        if improved:
                            break
                    if improved:
                        break
                if not improved:
                    break
            
            if not stage_improved and stage_idx > 0:
                break
    
    # Phase 3: Final polish with very small steps
    if _elapsed() < _DEADLINE - 3:
        polish_steps = [0.001, 0.0005, 0.0002]
        
        for step in polish_steps:
            if _elapsed() > _DEADLINE - 3:
                break
            
            improved = True
            while improved and _elapsed() < _DEADLINE - 3:
                improved = False
                for i in range(6):
                    for sign in [-1, 1]:
                        candidate = best_coeffs.copy()
                        candidate[i] += sign * step
                        if not all(-10 <= c <= 10 for c in candidate):
                            continue
                        
                        # Validate gamma range
                        if not _validate_all_models(candidate, test_models, a_vals):
                            continue
                        
                        # Use exact max error
                        try:
                            err = _compute_max_error(candidate, test_models, a_vals)
                        except:
                            continue
                        
                        if err < best_err - 1e-13:
                            best_err = err
                            best_coeffs = candidate
                            improved = True
                            break
                    if improved:
                        break
    
    # Final validation
    if _elapsed() > _DEADLINE - 2:
        return [0.55, 0.0, 0.0, 0.0, 0.0, 0.0]
    
    final_valid = _validate_all_models(best_coeffs, test_models, a_vals)
    
    if not final_valid:
        return [0.55, 0.0, 0.0, 0.0, 0.0, 0.0]
    
    return best_coeffs

# EVOLVE-BLOCK-END
