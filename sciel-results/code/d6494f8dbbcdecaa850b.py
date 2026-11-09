"""Optimized program for the uncertainty inequality task."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import time

_deadline = None
_best_result = None
_best_value = np.inf

def _check_deadline():
    global _deadline
    if _deadline is not None and time.time() > _deadline:
        return True
    return False

def _compute_c12(c0, c4, c8):
    """Compute c12 such that g(0) = 0."""
    H0_0 = 1.0
    H4_0 = 12.0
    H8_0 = 5040.0
    H12_0 = -47900160.0
    return -(c0 * H0_0 + c4 * H4_0 + c8 * H8_0) / H12_0

def _objective(coeffs):
    """Compute the upper bound on A(f)*A(Fourier(f))."""
    global _best_value, _best_result
    c0, c4, c8 = coeffs
    c12 = _compute_c12(c0, c4, c8)
    
    # Polynomial coefficients for p(x) = c0*H0 + c4*H4 + c8*H8 + c12*H12
    # Using physicists' Hermite polynomials
    p_coeffs = np.zeros(13)
    # H0 = 1
    p_coeffs[0] += c0
    # H4 = 16x^4 - 48x^2 + 12
    p_coeffs[0] += c4 * 12
    p_coeffs[2] += c4 * (-48)
    p_coeffs[4] += c4 * 16
    # H8 = 256x^8 - 4480x^6 + 23040x^4 - 26880x^2 + 5040
    p_coeffs[0] += c8 * 5040
    p_coeffs[2] += c8 * (-26880)
    p_coeffs[4] += c8 * 23040
    p_coeffs[6] += c8 * (-4480)
    p_coeffs[8] += c8 * 256
    # H12 = 4096x^12 - 163840x^10 + 1443840x^8 - 5506560x^6 + 6652800x^4 - 2540160x^2 + 1029600
    p_coeffs[0] += c12 * 1029600
    p_coeffs[2] += c12 * (-2540160)
    p_coeffs[4] += c12 * 6652800
    p_coeffs[6] += c12 * (-5506560)
    p_coeffs[8] += c12 * 1443840
    p_coeffs[10] += c12 * (-163840)
    p_coeffs[12] += c12 * 4096
    
    # g(x) = p(x) * exp(-x^2/2)
    # For numerical integration, use a grid
    n_pts = 8000
    x = np.linspace(-15, 15, n_pts)
    dx = x[1] - x[0]
    
    # Evaluate polynomial
    poly = np.zeros_like(x)
    for k in range(0, 13, 2):  # Only even powers
        if p_coeffs[k] != 0:
            poly += p_coeffs[k] * x**k
    
    g = poly * np.exp(-x**2 / 2)
    
    # L1 norm of g
    norm_g = np.sum(np.abs(g)) * dx
    
    # Compute Fourier transform numerically
    # FT{g}(xi) = integral g(x) cos(xi*x) dx (g is even)
    n_freq = 4000
    xi = np.linspace(-10, 10, n_freq)
    dxi = xi[1] - xi[0]
    
    # Use chunked computation to save memory
    ft_abs_sum = 0.0
    chunk_size = 1000
    for i_start in range(0, n_freq, chunk_size):
        i_end = min(i_start + chunk_size, n_freq)
        xi_chunk = xi[i_start:i_end]
        cos_mat = np.cos(np.outer(xi_chunk, x))
        ft_chunk = cos_mat @ g * dx
        ft_abs_sum += np.sum(np.abs(ft_chunk))
    
    norm_g_ft = ft_abs_sum * dxi
    
    # Upper bound is the product
    value = norm_g * norm_g_ft
    
    if value < _best_value:
        _best_value = value
        _best_result = np.array([c0, c4, c8])
    
    return value

def construct():
    global _deadline, _best_result, _best_value
    
    _deadline = time.time() + 175  # Safety margin
    _best_value = np.inf
    _best_result = np.array([-113.0/100, 1.0/25, 1.0/3240])
    
    # Evaluate seed
    seed_value = _objective(_best_result)
    
    # Local optimization from seed
    try:
        result = minimize(
            _objective,
            _best_result,
            method='Nelder-Mead',
            options={'maxiter': 1000, 'xatol': 1e-12, 'fatol': 1e-14, 'adaptive': True}
        )
        if _check_deadline():
            return _best_result
        if result.fun < _best_value:
            _best_value = result.fun
            _best_result = result.x
    except Exception:
        pass
    
    # Try multiple starting points
    start_points = [
        np.array([-1.13, 0.04, 0.000308642]),
        np.array([-1.0, 0.05, 0.0003]),
        np.array([-1.2, 0.03, 0.0004]),
        np.array([-1.1, 0.045, 0.00025]),
        np.array([-1.15, 0.035, 0.00035]),
        np.array([-1.05, 0.055, 0.0002]),
        np.array([-1.25, 0.025, 0.00045]),
        np.array([-0.95, 0.06, 0.0002]),
    ]
    
    for sp in start_points:
        if _check_deadline():
            break
        try:
            result = minimize(
                _objective,
                sp,
                method='Nelder-Mead',
                options={'maxiter': 500, 'xatol': 1e-12, 'fatol': 1e-14, 'adaptive': True}
            )
            if result.fun < _best_value:
                _best_value = result.fun
                _best_result = result.x
        except Exception:
            pass
    
    # Fine-tuning around the best found
    if not _check_deadline():
        best = _best_result.copy()
        for scale in [0.01, 0.001, 0.0001]:
            if _check_deadline():
                break
            n_grid = 5
            half = n_grid // 2
            for i in range(-half, half + 1):
                if _check_deadline():
                    break
                for j in range(-half, half + 1):
                    if _check_deadline():
                        break
                    for k in range(-half, half + 1):
                        if _check_deadline():
                            break
                        candidate = best + np.array([
                            i * scale, 
                            j * scale * 0.5, 
                            k * scale * 0.01
                        ])
                        try:
                            val = _objective(candidate)
                            if val < _best_value:
                                _best_value = val
                                _best_result = candidate
                        except Exception:
                            pass
    
    return _best_result
# EVOLVE-BLOCK-END
