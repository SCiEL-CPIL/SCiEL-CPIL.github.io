import numpy as np
import time
from scipy.optimize import minimize

# Precompute Hermite polynomials on a fixed grid for speed
_X_MIN = -15.0
_X_MAX = 15.0
_N_GRID = 10001
_x_grid = np.linspace(_X_MIN, _X_MAX, _N_GRID)
_dx = _x_grid[1] - _x_grid[0]
_exp_neg_half_x2 = np.exp(-_x_grid**2 / 2.0)

def _hermite_poly(n, x):
    """Compute the physicists' Hermite polynomial H_n(x)."""
    if n == 0:
        return np.ones_like(x)
    elif n == 1:
        return 2 * x
    else:
        h_prev = np.ones_like(x)
        h_curr = 2 * x
        for k in range(2, n + 1):
            h_next = 2 * x * h_curr - 2 * (k - 1) * h_prev
            h_prev = h_curr
            h_curr = h_next
        return h_curr

# Precompute the Hermite basis functions times Gaussian
_H0 = _hermite_poly(0, _x_grid) * _exp_neg_half_x2
_H4 = _hermite_poly(4, _x_grid) * _exp_neg_half_x2
_H8 = _hermite_poly(8, _x_grid) * _exp_neg_half_x2
_H12 = _hermite_poly(12, _x_grid) * _exp_neg_half_x2

# Precompute at x=0
_H0_0 = 1.0
_H4_0 = 12.0
_H8_0 = 1680.0
_H12_0 = 665280.0

def _compute_bound(a0, a4, a8):
    """
    Compute the uncertainty product upper bound for g(x) = (a0*H0 + a4*H4 + a8*H8 + a12*H12) * exp(-x^2/2)
    where a12 is determined by the constraint g(0) = 0.
    """
    a12 = -(a0 + 12.0 * a4 + 1680.0 * a8) / 665280.0
    
    g = a0 * _H0 + a4 * _H4 + a8 * _H8 + a12 * _H12
    
    l2_g_sq = np.dot(g, g) * _dx
    l2_g = np.sqrt(l2_g_sq)
    
    l4_integral = np.dot(g**2, g**2) * _dx
    norm_4_sq = l4_integral ** 0.5
    
    c4 = norm_4_sq / l2_g
    
    return c4

def construct():
    a0_init = -113.0 / 100.0
    a4_init = 1.0 / 25.0
    a8_init = 1.0 / 3240.0
    
    start_time = time.time()
    deadline = start_time + 175.0  # Safety margin before 180s limit
    
    best_coeffs = np.array([a0_init, a4_init, a8_init])
    best_val = _compute_bound(a0_init, a4_init, a8_init)
    
    rng = np.random.default_rng(42)
    
    # Phase 1: Multi-start Nelder-Mead
    n_starts = 30
    for i in range(n_starts):
        if time.time() > deadline - 40.0:
            break
        
        if i == 0:
            x0 = np.array([a0_init, a4_init, a8_init])
        else:
            # Adaptive scale: starts small, grows to explore wider
            scale = 0.1 + 0.9 * (i / max(1, n_starts - 1))
            x0 = np.array([a0_init, a4_init, a8_init]) + scale * rng.standard_normal(3)
        
        try:
            result = minimize(
                _compute_bound,
                x0,
                method='Nelder-Mead',
                options={'maxiter': 800, 'xatol': 1e-10, 'fatol': 1e-12, 'adaptive': True}
            )
            
            if result.fun < best_val:
                best_val = result.fun
                best_coeffs = result.x.copy()
        except Exception:
            continue
    
    # Phase 2: Local polish with Powell (derivative-free, robust)
    if time.time() < deadline - 20.0:
        try:
            result = minimize(
                _compute_bound,
                best_coeffs,
                method='Powell',
                options={'maxiter': 500, 'xtol': 1e-12, 'ftol': 1e-14}
            )
            if result.fun < best_val:
                best_val = result.fun
                best_coeffs = result.x.copy()
        except Exception:
            pass
    
    # Phase 3: Final high-precision Nelder-Mead polish
    if time.time() < deadline - 5.0:
        try:
            result = minimize(
                _compute_bound,
                best_coeffs,
                method='Nelder-Mead',
                options={'maxiter': 2000, 'xatol': 1e-14, 'fatol': 1e-16, 'adaptive': True}
            )
            if result.fun < best_val:
                best_val = result.fun
                best_coeffs = result.x.copy()
        except Exception:
            pass
    
    return best_coeffs
