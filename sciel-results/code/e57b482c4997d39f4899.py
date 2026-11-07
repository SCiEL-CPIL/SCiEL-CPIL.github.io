"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize, brentq
import time


# Hermite polynomial coefficients (physicists' convention): H_n(x) = sum_k a[k] x^(n-2k)
# H_0 = 1
# H_4 = 16x^4 - 48x^2 + 12
# H_8 = 256x^8 - 1344x^6 + 2688x^4 - 1344x^2 + 112
# H_12 = 4096x^12 - 46080x^10 + 302400x^8 - 806400x^6 + 1008000x^4 - 604800x^2 + 10368

# g(x) = c0*H0 + c1*H4 + c2*H8 + c3*H12
# g(0) = c0*1 + c1*12 + c2*112 + c3*10368 = 0
# => c3 = -(c0 + 12*c1 + 112*c2) / 10368

# q(x) = g(x) / x^2. Since g(0)=0 and g is even, g(x) = x^2 * q(x) where q is even polynomial of degree 10.
# q(x) = a12*x^10 + a10*x^8 + a8*x^6 + a6*x^4 + a4*x^2 + a2
# where:
# a12 = 4096*c3
# a10 = -46080*c3 + 256*c2
# a8  = 302400*c3 - 1344*c2 - 48*c1
# a6  = -806400*c3 + 2688*c2 - 16*c1 + c0
# a4  = 1008000*c3 - 1344*c2
# a2  = -604800*c3 + 112*c2 + 12*c1

def _build_q_coeffs(c0, c1, c2):
    c3 = -(c0 + 12.0 * c1 + 112.0 * c2) / 10368.0
    a12 = 4096.0 * c3
    a10 = -46080.0 * c3 + 256.0 * c2
    a8  = 302400.0 * c3 - 1344.0 * c2 - 48.0 * c1
    a6  = -806400.0 * c3 + 2688.0 * c2 - 16.0 * c1 + c0
    a4  = 1008000.0 * c3 - 1344.0 * c2
    a2  = -604800.0 * c3 + 112.0 * c2 + 12.0 * c1
    return np.array([a12, a10, a8, a6, a4, a2])


def _q_eval(coeffs, x):
    """Evaluate q(x) = a12*x^10 + a10*x^8 + a8*x^6 + a6*x^4 + a4*x^2 + a2"""
    # Use Horner's method in z = x^2
    z = x * x
    a12, a10, a8, a6, a4, a2 = coeffs
    return ((a12 * z + a10) * z + a8) * z * z + (a6 * z + a4) * z + a2


def _q_eval_vec(coeffs, x):
    """Vectorized evaluation of q(x)."""
    z = x * x
    a12, a10, a8, a6, a4, a2 = coeffs
    return ((a12 * z + a10) * z + a8) * z * z + (a6 * z + a4) * z + a2


def _find_largest_positive_sign_change_root(coeffs):
    """Find the largest positive x where q(x) changes sign (crosses zero).
    We look for the largest positive simple root of q.
    """
    a12, a10, a8, a6, a4, a2 = coeffs
    
    # Build the polynomial in z = x^2: p(z) = a12*z^5 + a10*z^4 + a8*z^3 + a6*z^2 + a4*z + a2
    # We need positive z roots where p(z) changes sign, then r = sqrt(z)
    
    # Use numpy's roots for the polynomial in z
    # p(z) = a12*z^5 + a10*z^4 + a8*z^3 + a6*z^2 + a4*z + a2
    p_coeffs = np.array([a12, a10, a8, a6, a4, a2])
    
    # Remove leading zeros
    while len(p_coeffs) > 1 and abs(p_coeffs[0]) < 1e-30:
        p_coeffs = p_coeffs[1:]
    
    if len(p_coeffs) == 1:
        # Constant polynomial, no roots
        return None
    
    try:
        z_roots = np.roots(p_coeffs)
    except Exception:
        return None
    
    # Filter for positive real roots (z > 0)
    # A root changes sign if it has odd multiplicity
    # For our purposes, we'll find all positive real roots and check sign change
    
    positive_real_roots = []
    for zr in z_roots:
        if abs(zr.imag) < 1e-10 and zr.real > 1e-12:
            positive_real_roots.append(zr.real)
    
    if not positive_real_roots:
        return None
    
    # For each positive real root z, check if q changes sign there
    # q(x) = p(x^2), so at x = sqrt(z), q changes sign iff p changes sign at z
    # Check by evaluating p at z ± epsilon
    eps = 1e-8
    sign_changing_z = []
    for z in positive_real_roots:
        p_minus = _q_eval(coeffs, np.sqrt(z - eps)) if z > eps else _q_eval(coeffs, 0)
        p_plus = _q_eval(coeffs, np.sqrt(z + eps))
        if p_minus * p_plus < 0:
            sign_changing_z.append(z)
    
    if not sign_changing_z:
        return None
    
    # The largest r = sqrt(z_max)
    z_max = max(sign_changing_z)
    return np.sqrt(z_max)


def _objective(params):
    """Objective: r^2 / (2*pi) where r is the largest positive sign-changing root of q.
    Lower is better.
    """
    c0, c1, c2 = params
    coeffs = _build_q_coeffs(c0, c1, c2)
    r = _find_largest_positive_sign_change_root(coeffs)
    if r is None:
        return 1e10  # Penalty
    return r * r / (2.0 * np.pi)


def _objective_smooth(params, temperature=1e-6):
    """Smooth surrogate for the objective using a soft-min approach.
    We approximate the max root by considering all positive real roots.
    """
    c0, c1, c2 = params
    coeffs = _build_q_coeffs(c0, c1, c2)
    a12, a10, a8, a6, a4, a2 = coeffs
    
    p_coeffs = np.array([a12, a10, a8, a6, a4, a2])
    while len(p_coeffs) > 1 and abs(p_coeffs[0]) < 1e-30:
        p_coeffs = p_coeffs[1:]
    
    if len(p_coeffs) == 1:
        return 1e10
    
    try:
        z_roots = np.roots(p_coeffs)
    except Exception:
        return 1e10
    
    # Get positive real roots
    positive_z = []
    for zr in z_roots:
        if abs(zr.imag) < 1e-8 and zr.real > 1e-10:
            positive_z.append(zr.real)
    
    if not positive_z:
        return 1e10
    
    # Use log-sum-exp for soft-max of z values (since r^2 = z)
    # value = z_max / (2*pi), so we want to minimize z_max
    # Soft-max: z_soft = log(sum(exp(z_i / T))) * T
    z_arr = np.array(positive_z)
    z_max = np.max(z_arr)
    # Soft-max approximation
    z_soft = z_max + temperature * np.log(np.sum(np.exp((z_arr - z_max) / temperature)))
    return z_soft / (2.0 * np.pi)


def construct():
    start_time = time.time()
    deadline = start_time + 175.0  # Safety margin before 210s kill
    
    best_params = np.array([-113.0 / 100.0, 1.0 / 25.0, 1.0 / 3240.0])
    best_val = _objective(best_params)
    
    if best_val > 1e9:
        best_val = 1e10
    
    rng = np.random.RandomState(42)
    
    # Phase 1: Broad exploration with diverse starts
    starts = [best_params.copy()]
    
    # Random perturbations at various scales
    for scale in [0.01, 0.05, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0]:
        for _ in range(8):
            p = best_params + scale * rng.normal(0, 1.0, size=3)
            starts.append(p)
    
    # Coordinate-wise perturbations
    for i in range(3):
        for delta in [0.01, 0.05, 0.1, 0.5, 1.0]:
            for sign in [1, -1]:
                p = best_params.copy()
                p[i] += sign * delta
                starts.append(p)
    
    # Sort by initial objective
    starts_with_obj = []
    for s in starts:
        v = _objective(s)
        starts_with_obj.append((v, s))
    starts_with_obj.sort(key=lambda x: x[0])
    
    # Phase 2: Local optimization from top starts
    n_local = min(20, len(starts_with_obj))
    for i in range(n_local):
        if time.time() > deadline - 40.0:
            break
        
        init_val, init_params = starts_with_obj[i]
        if init_val > best_val * 1.5:
            continue
        
        try:
            result = minimize(_objective, init_params, method='Nelder-Mead',
                             options={'maxiter': 3000, 'xatol': 1e-10, 'fatol': 1e-12,
                                     'adaptive': True})
            
            if result.fun < best_val:
                best_val = result.fun
                best_params = result.x.copy()
        except Exception:
            continue
    
    # Phase 3: Polish with tighter tolerances
    if time.time() < deadline - 15.0:
        try:
            result = minimize(_objective, best_params, method='Nelder-Mead',
                             options={'maxiter': 10000, 'xatol': 1e-13, 'fatol': 1e-15,
                                     'adaptive': True})
            if result.fun < best_val:
                best_val = result.fun
                best_params = result.x.copy()
        except Exception:
            pass
    
    # Phase 4: Final high-precision polish
    if time.time() < deadline - 5.0:
        try:
            result = minimize(_objective, best_params, method='Nelder-Mead',
                             options={'maxiter': 5000, 'xatol': 1e-14, 'fatol': 1e-16,
                                     'adaptive': True})
            if result.fun < best_val:
                best_val = result.fun
                best_params = result.x.copy()
        except Exception:
            pass
    
    # Final verification
    final_val = _objective(best_params)
    if final_val < best_val:
        best_val = final_val
    
    return np.array(best_params)
# EVOLVE-BLOCK-END
