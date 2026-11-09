"""
Uncertainty inequality: tune Hermite-Gaussian test function coefficients to minimize the upper bound on C4.

Uses exact L2 norms via Hermite orthogonality and dense numerical sampling for L-infinity norms.
The key structural insight from the inspirations: for even-degree Hermite polynomials,
(-i)^n = 1 for n = 0, 4, 8, 12, so the Fourier transform of g has the SAME polynomial
combination as g itself. This means ||F(g)||_inf = ||g||_inf and the bound simplifies
to (||g||_inf / ||g||_2)^2.

We use a fast vectorized evaluation with precomputed grids, and a multi-phase
optimization strategy: coarse grid, Nelder-Mead refinement, and multi-start.
"""
import time
import math
import numpy as np
from scipy.special import eval_hermite
from scipy.optimize import minimize

DEADLINE_FRACTION = 0.88

# Precompute exact L2 norm coefficients
# ||H_n * exp(-x^2/2)||_2^2 = sqrt(pi) * 2^n * n!
_SQRT_PI = math.sqrt(math.pi)
_L2_COEF = {
    0: _SQRT_PI * 1.0,
    4: _SQRT_PI * (2**4) * math.factorial(4),
    8: _SQRT_PI * (2**8) * math.factorial(8),
    12: _SQRT_PI * (2**12) * math.factorial(12),
}

# Dense grid for L-infinity computation
_N_GRID = 100000
_X_FINE = np.linspace(-12, 12, _N_GRID)
_H0_F = np.ones_like(_X_FINE)
_H4_F = eval_hermite(4, _X_FINE)
_H8_F = eval_hermite(8, _X_FINE)
_H12_F = eval_hermite(12, _X_FINE)
_GAUSS_F = np.exp(-0.5 * _X_FINE**2)


def _compute_bound_fast(a0, a4, a8):
    """
    Compute the upper bound A(g)*A(g_hat) for g(x) = (a0*H0 + a4*H4 + a8*H8 + a12*H12)*exp(-x^2/2)
    where a12 is derived from g(0) = 0.
    
    Key insight: for even n, (-i)^n = 1, so F(g) has the same polynomial structure as g.
    Therefore ||F(g)||_inf = ||g||_inf and the bound = (||g||_inf / ||g||_2)^2.
    """
    # H0(0)=1, H4(0)=-3, H8(0)=105, H12(0)=-10395
    # g(0) = (a0*1 + a4*(-3) + a8*105 + a12*(-10395)) * 1 = 0
    a12 = (a0 - 3.0*a4 + 105.0*a8) / 10395.0
    
    # Exact L2 norm of g: ||g||_2^2 = sum a_k^2 * <H_k, H_k>
    norm2_g_sq = (a0**2 * _L2_COEF[0] + a4**2 * _L2_COEF[4] + 
                  a8**2 * _L2_COEF[8] + a12**2 * _L2_COEF[12])
    
    if norm2_g_sq <= 0:
        return 1e10
    
    norm2_g = np.sqrt(norm2_g_sq)
    
    # Compute L-infinity of g using vectorized operations
    poly_g = a0*_H0_F + a4*_H4_F + a8*_H8_F + a12*_H12_F
    g = poly_g * _GAUSS_F
    max_g = np.max(np.abs(g))
    
    # Since F(g) has the same structure (even degrees only), bound = (max_g / norm2_g)^2
    bound = (max_g / norm2_g) ** 2
    
    return bound


def _refine_linf(a0, a4, a8):
    """Refine the L-infinity norm using local minimization around the peak."""
    a12 = (a0 - 3.0*a4 + 105.0*a8) / 10395.0
    
    def f_abs(x):
        h0 = 1.0
        h4 = eval_hermite(4, x)
        h8 = eval_hermite(8, x)
        h12 = eval_hermite(12, x)
        poly = a0*h0 + a4*h4 + a8*h8 + a12*h12
        return abs(poly * np.exp(-0.5*x**2))
    
    # Find coarse peak
    poly_coarse = a0*_H0_F + a4*_H4_F + a8*_H8_F + a12*_H12_F
    g_coarse = poly_coarse * _GAUSS_F
    idx = np.argmax(np.abs(g_coarse))
    x_coarse = _X_FINE[idx]
    
    # Local refinement around the peak
    from scipy.optimize import minimize_scalar
    result = minimize_scalar(lambda x: -f_abs(x), bounds=(x_coarse - 0.05, x_coarse + 0.05), method='bounded', options={'xatol': 1e-14})
    return f_abs(result.x)


def _compute_bound_refined(a0, a4, a8):
    """Compute bound with refined L-infinity norm."""
    a12 = (a0 - 3.0*a4 + 105.0*a8) / 10395.0
    
    norm2_g_sq = (a0**2 * _L2_COEF[0] + a4**2 * _L2_COEF[4] + 
                  a8**2 * _L2_COEF[8] + a12**2 * _L2_COEF[12])
    
    if norm2_g_sq <= 0:
        return 1e10
    
    norm2_g = np.sqrt(norm2_g_sq)
    
    # Refined L-infinity
    max_g = _refine_linf(a0, a4, a8)
    
    bound = (max_g / norm2_g) ** 2
    return bound


def construct():
    start_time = time.time()
    deadline = 210.0 * DEADLINE_FRACTION  # ~185s
    
    best_val = float('inf')
    best_x = np.array([-113.0/100.0, 1.0/25.0, 1.0/3240.0])  # seed
    
    # Evaluate seed
    try:
        seed_val = _compute_bound_fast(best_x[0], best_x[1], best_x[2])
        if seed_val < best_val:
            best_val = seed_val
    except Exception:
        pass
    
    elapsed = time.time() - start_time
    if elapsed > deadline - 60:
        return best_x
    
    # Phase 1: Coarse grid search
    try:
        a0_center, a4_center, a8_center = best_x
        a0_span, a4_span, a8_span = 2.0, 0.2, 0.005
        
        n_a0, n_a4, n_a8 = 10, 8, 6
        
        a0_vals = np.linspace(a0_center - a0_span, a0_center + a0_span, n_a0)
        a4_vals = np.linspace(a4_center - a4_span, a4_center + a4_span, n_a4)
        a8_vals = np.linspace(a8_center - a8_span, a8_center + a8_span, n_a8)
        
        for a0 in a0_vals:
            for a4 in a4_vals:
                for a8 in a8_vals:
                    if time.time() - start_time > deadline - 50:
                        raise TimeoutError
                    try:
                        v = _compute_bound_fast(a0, a4, a8)
                        if v < best_val:
                            best_val = v
                            best_x = np.array([a0, a4, a8])
                    except Exception:
                        pass
    except Exception:
        pass
    
    elapsed = time.time() - start_time
    if elapsed > deadline - 40:
        return best_x
    
    # Phase 2: Nelder-Mead refinement
    try:
        result = minimize(
            lambda x: _compute_bound_fast(x[0], x[1], x[2]),
            best_x,
            method='Nelder-Mead',
            options={'maxiter': 2000, 'xatol': 1e-12, 'fatol': 1e-14, 'adaptive': True}
        )
        if result.fun < best_val:
            best_val = result.fun
            best_x = result.x
    except Exception:
        pass
    
    elapsed = time.time() - start_time
    if elapsed > deadline - 30:
        return best_x
    
    # Phase 3: Multi-start with different initializations
    np.random.seed(42)
    try:
        for i in range(10):
            if time.time() - start_time > deadline - 20:
                break
            scale = 0.02
            x0 = best_x + scale * np.random.randn(3)
            result = minimize(
                lambda x: _compute_bound_fast(x[0], x[1], x[2]),
                x0,
                method='Nelder-Mead',
                options={'maxiter': 1000, 'xatol': 1e-13, 'fatol': 1e-15, 'adaptive': True}
            )
            if result.fun < best_val:
                best_val = result.fun
                best_x = result.x
    except Exception:
        pass
    
    elapsed = time.time() - start_time
    if elapsed > deadline - 15:
        return best_x
    
    # Phase 4: High-precision polish with refined L-infinity
    try:
        def obj_refined(x):
            return _compute_bound_refined(x[0], x[1], x[2])
        
        result = minimize(
            obj_refined,
            best_x,
            method='Nelder-Mead',
            options={'maxiter': 500, 'xatol': 1e-14, 'fatol': 1e-16, 'adaptive': True}
        )
        if result.fun < best_val:
            best_val = result.fun
            best_x = result.x
    except Exception:
        pass
    
    return best_x
