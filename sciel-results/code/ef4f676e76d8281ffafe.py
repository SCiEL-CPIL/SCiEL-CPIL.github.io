"""Optimize uncertainty-inequality constant via smooth surrogate + multi-start."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize, differential_evolution
from scipy.optimize import brentq
import time
import sys

def _compute_q_coeffs(c0, c1, c2):
    """Compute coefficients of q(x) = g(x)/x^2 as polynomial in x^2 (degree 5 in x^2).
    Returns [a5, a4, a3, a2, a1, a0] where q = a5*y^5 + a4*y^4 + a3*y^3 + a2*y^2 + a1*y + a0, y=x^2
    """
    c3 = -(c0 + 3*c1 + 15*c2) / 105.0
    
    a5 = c3
    a4 = -60*c3
    a3 = 900*c3 + c2
    a2 = -3600*c3 - 24*c2
    a1 = 2880*c3 + 72*c2 + c1
    a0 = -432*c3 - 32*c2 - 6*c1
    
    return np.array([a5, a4, a3, a2, a1, a0])

def _eval_q_y(y, coeffs):
    """Evaluate q at y = x^2 using Horner's method."""
    a5, a4, a3, a2, a1, a0 = coeffs
    return (((((a5*y + a4)*y + a3)*y + a2)*y + a1)*y + a0)

def _eval_q_x(x, coeffs):
    """Evaluate q(x) = q_y(x^2)."""
    return _eval_q_y(x*x, coeffs)

def _find_largest_sign_changing_root(coeffs):
    """Find the largest real root of q where sign changes.
    coeffs: [a5, a4, a3, a2, a1, a0] for q(y) in terms of y=x^2.
    Returns r (the x-value, positive) or None.
    """
    # Find roots of q(y) = a5*y^5 + a4*y^4 + a3*y^3 + a2*y^2 + a1*y + a0
    # Only consider y >= 0 (since y = x^2)
    y_roots = np.roots(coeffs)
    
    # Filter real non-negative roots
    valid_y = []
    for yr in y_roots:
        if abs(yr.imag) < 1e-9 and yr.real > -1e-12:
            valid_y.append(max(0.0, yr.real))
    
    if not valid_y:
        return None
    
    valid_y.sort(reverse=True)
    
    # For each y-root, check if q changes sign (i.e., it's a simple root or odd multiplicity)
    # Check sign change by evaluating at y-eps and y+eps
    for y in valid_y:
        eps = 1e-8 * max(1.0, abs(y))
        q_minus = _eval_q_y(max(0, y - eps), coeffs)
        q_plus = _eval_q_y(y + eps, coeffs)
        if q_minus * q_plus < 0:
            return np.sqrt(y)
    
    return None

def _objective(params):
    """Objective: minimize r^2 where r is largest sign-changing root."""
    c0, c1, c2 = params
    coeffs = _compute_q_coeffs(c0, c1, c2)
    r = _find_largest_sign_changing_root(coeffs)
    if r is None:
        return 1e10
    return r * r

def _objective_smooth(params, beta=100.0):
    """Smooth surrogate: use log-sum-exp over all real roots to approximate max root.
    This handles the nonsmoothness of 'largest sign-changing root'.
    """
    c0, c1, c2 = params
    coeffs = _compute_q_coeffs(c0, c1, c2)
    
    # Find all real roots in y-space
    y_roots = np.roots(coeffs)
    
    # Get real non-negative roots
    real_ys = []
    for yr in y_roots:
        if abs(yr.imag) < 1e-9 and yr.real > -1e-12:
            real_ys.append(max(0.0, yr.real))
    
    if not real_ys:
        return 1e10
    
    # For each root, check if it's a sign-changing root (odd multiplicity)
    sign_change_ys = []
    for y in real_ys:
        eps = 1e-8 * max(1.0, abs(y))
        q_minus = _eval_q_y(max(0, y - eps), coeffs)
        q_plus = _eval_q_y(y + eps, coeffs)
        if q_minus * q_plus < 0:
            sign_change_ys.append(y)
    
    if not sign_change_ys:
        return 1e10
    
    # Use soft-max (log-sum-exp) to smoothly approximate max y
    ys = np.array(sign_change_ys)
    max_y = np.max(ys)
    # Log-sum-exp for numerical stability
    shifted = ys - max_y
    log_sum_exp = max_y + np.log(np.sum(np.exp(shifted)))
    # soft_max_y = log_sum_exp / beta (this approximates max_y as beta -> inf)
    soft_max_y = log_sum_exp / beta
    
    return soft_max_y

def _polish(params, max_iter=1000):
    """High-precision polish using Nelder-Mead with tight tolerances."""
    try:
        result = minimize(_objective, params, method='Nelder-Mead',
                         options={'maxiter': max_iter, 'xatol': 1e-14, 'fatol': 1e-16,
                                  'adaptive': True})
        return result.x, result.fun
    except Exception:
        return params, _objective(params)

def construct():
    start_time = time.time()
    deadline = 205.0  # safety margin before 210s limit
    
    def time_left():
        return deadline - time.time()
    
    # Seed
    seed = np.array([-113.0/100.0, 1.0/25.0, 1.0/3240.0])
    best_params = seed.copy()
    best_value = _objective(seed)
    
    rng = np.random.RandomState(42)
    
    # Phase 1: Broad exploration with differential evolution
    if time_left() > 60:
        try:
            bounds = [(-5, 5), (-5, 5), (-5, 5)]
            de_result = differential_evolution(_objective, bounds, maxiter=50,
                                              popsize=15, tol=1e-10, seed=42,
                                              polish=False, init='latinhypercube')
            if de_result.fun < best_value:
                best_value = de_result.fun
                best_params = de_result.x.copy()
        except Exception:
            pass
    
    # Phase 2: Multi-start local optimization
    n_starts = 80
    for i in range(n_starts):
        if time_left() < 15:
            break
        
        if i == 0:
            x0 = best_params.copy()
        elif i == 1:
            x0 = seed.copy()
        elif i < n_starts // 3:
            # Perturb around best
            x0 = best_params + rng.randn(3) * 0.05
        elif i < 2 * n_starts // 3:
            # Perturb around seed
            x0 = seed + rng.randn(3) * 0.2
        else:
            # Uniform random in reasonable range
            x0 = rng.uniform([-3, -3, -3], [3, 3, 3])
        
        try:
            # First try smooth objective for better convergence
            result_smooth = minimize(_objective_smooth, x0, method='Nelder-Mead',
                                     options={'maxiter': 300, 'xatol': 1e-10, 'fatol': 1e-12})
            
            # Then refine with exact objective
            result = minimize(_objective, result_smooth.x, method='Nelder-Mead',
                             options={'maxiter': 500, 'xatol': 1e-12, 'fatol': 1e-14,
                                      'adaptive': True})
            
            if result.fun < best_value:
                best_value = result.fun
                best_params = result.x.copy()
        except Exception:
            pass
    
    # Phase 3: Final high-precision polish
    if time_left() > 5:
        polished_params, polished_val = _polish(best_params, max_iter=2000)
        if polished_val < best_value:
            best_value = polished_val
            best_params = polished_params
    
    # Verify and return
    final_val = _objective(best_params)
    if final_val < best_value:
        best_value = final_val
    
    return best_params
# EVOLVE-BLOCK-END
