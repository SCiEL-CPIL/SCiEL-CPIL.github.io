"""Uncertainty bound minimization via stationary point search (root-finding)."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import root, minimize
from math import factorial, sqrt, pi

_DEADLINE = 190.0


def _hermite_coeffs(n):
    """Return coefficients of the physicists' Hermite polynomial H_n(x) as dict power -> coeff."""
    if n == 0:
        return {0: 1.0}
    if n == 1:
        return {1: 2.0}
    h_prev = {0: 1.0}
    h_curr = {1: 2.0}
    for m in range(2, n + 1):
        h_new = {}
        for p, c in h_curr.items():
            new_p = p + 1
            h_new[new_p] = h_new.get(new_p, 0.0) + 2.0 * c
        for p, c in h_prev.items():
            h_new[p] = h_new.get(p, 0.0) - 2.0 * (m - 1) * c
        h_new = {p: c for p, c in h_new.items() if abs(c) > 1e-15}
        h_prev, h_curr = h_curr, h_new
    return h_curr


def _int_x2m(m):
    """∫_{-∞}^{∞} x^{2m} e^{-x^2} dx = sqrt(π) (2m-1)!! / 2^m"""
    if m == 0:
        return sqrt(pi)
    val = sqrt(pi)
    for j in range(1, m + 1):
        val *= (2 * j - 1) / 2.0
    return val


# Precompute Hermite coefficient dicts for degrees 0, 4, 8, 12
_H0 = _hermite_coeffs(0)
_H4 = _hermite_coeffs(4)
_H8 = _hermite_coeffs(8)
_H12 = _hermite_coeffs(12)

# Precompute p_poly basis: for each degree, store the contribution to p_poly
# p_poly has length 13 (powers 0..12)
_DEG_MAX = 12
_POLY_LEN = _DEG_MAX + 1

# For each of the 4 Hermite polynomials, precompute the coefficient vector
_H_COEFFS = []
for _hc in [_H0, _H4, _H8, _H12]:
    vec = np.zeros(_POLY_LEN)
    for p, c in _hc.items():
        if p < _POLY_LEN:
            vec[p] = c
    _H_COEFFS.append(vec)
_H_COEFFS = np.array(_H_COEFFS)  # shape (4, 13)

# H values at zero for the constraint
_H0_0 = 1.0
_H4_0 = 12.0   # (-1)^2 * 4!/2!
_H8_0 = 2520.0  # (-1)^4 * 8!/4!
_H12_0 = 27720.0  # (-1)^6 * 12!/6!


def _compute_moments(a0, a4, a8):
    """Compute the uncertainty bound given coefficients a0, a4, a8."""
    a12 = -(a0 * _H0_0 + a4 * _H4_0 + a8 * _H8_0) / _H12_0
    
    # Build p_poly = a0*H0 + a4*H4 + a8*H8 + a12*H12
    coeffs = np.array([a0, a4, a8, a12])
    p_poly = coeffs @ _H_COEFFS  # shape (13,)
    
    # p(x)^2 as polynomial
    p2_poly = np.zeros(2 * _DEG_MAX + 1)
    for i in range(_POLY_LEN):
        if p_poly[i] != 0.0:
            for j in range(_POLY_LEN):
                p2_poly[i + j] += p_poly[i] * p_poly[j]
    
    # M0 = ∫ p^2 e^{-x^2} dx
    M0 = 0.0
    for k in range(0, 2 * _DEG_MAX + 1, 2):
        m = k // 2
        M0 += p2_poly[k] * _int_x2m(m)
    
    # M2 = ∫ x^2 p^2 e^{-x^2} dx
    M2 = 0.0
    for k in range(0, 2 * _DEG_MAX + 1, 2):
        m_plus_1 = (k + 2) // 2
        M2 += p2_poly[k] * _int_x2m(m_plus_1)
    
    # g'(x) = (p' - x p) e^{-x^2/2}
    # p'(x)
    p_prime = np.zeros(_POLY_LEN - 1)
    for i in range(1, _POLY_LEN):
        p_prime[i - 1] = p_poly[i] * i
    
    # (p')^2
    pprime_sq = np.zeros(2 * (_POLY_LEN - 1) - 1)
    for i in range(_POLY_LEN - 1):
        if p_prime[i] != 0.0:
            for j in range(_POLY_LEN - 1):
                pprime_sq[i + j] += p_prime[i] * p_prime[j]
    
    int_pprime_sq = 0.0
    for k in range(0, len(pprime_sq), 2):
        m = k // 2
        int_pprime_sq += pprime_sq[k] * _int_x2m(m)
    
    # x*p(x)
    xp = np.zeros(_POLY_LEN + 1)
    for i in range(_POLY_LEN):
        xp[i + 1] = p_poly[i]
    
    # x*p(x) * p'(x)
    xppp = np.zeros(_POLY_LEN + (_POLY_LEN - 1))
    for i in range(_POLY_LEN + 1):
        if xp[i] != 0.0:
            for j in range(_POLY_LEN - 1):
                xppp[i + j] += xp[i] * p_prime[j]
    
    int_xppp = 0.0
    for k in range(0, len(xppp), 2):
        m = k // 2
        int_xppp += xppp[k] * _int_x2m(m)
    
    int_gprime_sq = int_pprime_sq - 2.0 * int_xppp + M2
    
    if M0 <= 1e-15 or int_gprime_sq <= 1e-15:
        return np.inf
    
    bound = np.sqrt(M2 * int_gprime_sq) / M0
    return bound


def _gradient(params):
    """
    Compute gradient of the bound with respect to [a0, a4, a8].
    Uses central finite differences for stability and simplicity, 
    which is sufficient for 3D root finding.
    """
    a0, a4, a8 = params
    h = 1e-6
    grad = np.zeros(3)
    
    # d/d a0
    f_plus = _compute_moments(a0 + h, a4, a8)
    f_minus = _compute_moments(a0 - h, a4, a8)
    if np.isfinite(f_plus) and np.isfinite(f_minus):
        grad[0] = (f_plus - f_minus) / (2 * h)
    else:
        # Fallback to forward difference if inf
        f_0 = _compute_moments(a0, a4, a8)
        if np.isfinite(f_plus) and np.isfinite(f_0):
            grad[0] = (f_plus - f_0) / h
        else:
            grad[0] = 0.0

    # d/d a4
    f_plus = _compute_moments(a0, a4 + h, a8)
    f_minus = _compute_moments(a0, a4 - h, a8)
    if np.isfinite(f_plus) and np.isfinite(f_minus):
        grad[1] = (f_plus - f_minus) / (2 * h)
    else:
        f_0 = _compute_moments(a0, a4, a8)
        if np.isfinite(f_plus) and np.isfinite(f_0):
            grad[1] = (f_plus - f_0) / h
        else:
            grad[1] = 0.0

    # d/d a8
    f_plus = _compute_moments(a0, a4, a8 + h)
    f_minus = _compute_moments(a0, a4, a8 - h)
    if np.isfinite(f_plus) and np.isfinite(f_minus):
        grad[2] = (f_plus - f_minus) / (2 * h)
    else:
        f_0 = _compute_moments(a0, a4, a8)
        if np.isfinite(f_plus) and np.isfinite(f_0):
            grad[2] = (f_plus - f_0) / h
        else:
            grad[2] = 0.0
            
    return grad


def construct():
    """
    Find coefficients (a0, a4, a8) that minimize the uncertainty bound.
    Returns np.array([a0, a4, a8]).
    """
    start_time = time.time()
    
    # Known good solution in linear space
    a0_lin = -113.0 / 100
    a4_lin = 1.0 / 25
    a8_lin = 1.0 / 3240
    
    best_x = np.array([a0_lin, a4_lin, a8_lin])
    best_bound = _compute_moments(a0_lin, a4_lin, a8_lin)
    
    # Generate diverse starting points
    # 1. The known good solution
    starts = [best_x.copy()]
    
    # 2. Random perturbations around the known solution
    rng = np.random.default_rng(42)
    for _ in range(50):
        noise = rng.normal(0, 0.1, size=3)
        x_start = best_x + noise
        starts.append(x_start)
        
    # 3. Random points in a reasonable range
    # Based on the magnitudes of the known solution: a0 ~ -1, a4 ~ 0.04, a8 ~ 0.0003
    # We explore a box around these values
    for _ in range(50):
        a0 = rng.uniform(-2.0, 0.5)
        a4 = rng.uniform(-0.1, 0.2)
        a8 = rng.uniform(-0.001, 0.002)
        x_start = np.array([a0, a4, a8])
        starts.append(x_start)
        
    # 4. Some extreme points to check boundaries
    starts.append(np.array([-5.0, 0.1, 0.0]))
    starts.append(np.array([0.0, 0.1, 0.0]))
    starts.append(np.array([0.0, 0.0, 0.001]))
    
    # Use root-finding to find stationary points from each start
    for x_start in starts:
        if time.time() - start_time > _DEADLINE - 30:
            break
            
        try:
            # Find a root of the gradient
            res = root(_gradient, x_start, method='hybr', tol=1e-12, maxfev=1000)
            if res.success:
                x_root = res.x
                b_root = _compute_moments(x_root[0], x_root[1], x_root[2])
                if np.isfinite(b_root) and b_root < best_bound:
                    best_bound = b_root
                    best_x = x_root.copy()
        except Exception:
            pass
        
        # Also try a direct minimization from this start to catch cases where root-finding fails
        if time.time() - start_time > _DEADLINE - 20:
            break
        try:
            res_min = minimize(
                lambda p: _compute_moments(p[0], p[1], p[2]),
                x_start,
                method='Nelder-Mead',
                options={'maxiter': 200, 'xatol': 1e-10, 'fatol': 1e-12}
            )
            b_min = res_min.fun
            if np.isfinite(b_min) and b_min < best_bound:
                best_bound = b_min
                best_x = res_min.x.copy()
        except Exception:
            pass

    # Final refinement of the best point found using L-BFGS-B with gradient
    if time.time() - start_time < _DEADLINE - 10:
        try:
            # Define objective and gradient for L-BFGS-B
            def obj_grad(p):
                val = _compute_moments(p[0], p[1], p[2])
                grad = _gradient(p)
                return val, grad
                
            res_final = minimize(
                obj_grad,
                best_x,
                jac=True,
                method='L-BFGS-B',
                options={'maxiter': 100, 'ftol': 1e-15, 'gtol': 1e-12}
            )
            b_final = res_final.fun
            if np.isfinite(b_final) and b_final < best_bound:
                best_bound = b_final
                best_x = res_final.x.copy()
        except Exception:
            pass

    return best_x
# EVOLVE-BLOCK-END
