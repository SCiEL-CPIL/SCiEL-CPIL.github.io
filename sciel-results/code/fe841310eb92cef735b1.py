"""Optimized program for uncertainty inequality - minimize C4 bound."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize, differential_evolution
import time

def _hermite_values(x, n):
    """Compute physicist's Hermite polynomial H_n(x) at point(s) x."""
    if n == 0:
        return np.ones_like(x) if np.ndim(x) else 1.0
    elif n == 1:
        return 2 * x
    elif n == 2:
        return 4 * x**2 - 2
    elif n == 4:
        return 16 * x**4 - 48 * x**2 + 12
    elif n == 8:
        return 256 * x**8 - 1792 * x**6 + 3360 * x**4 - 1680 * x**2 + 168
    elif n == 12:
        return (4096 * x**12 - 46080 * x**10 + 241920 * x**8 
                - 483840 * x**6 + 332640 * x**4 - 70560 * x**2 + 10395)
    else:
        raise ValueError(f"Unsupported Hermite degree: {n}")

def _compute_c4_bound(coeffs):
    """
    Compute the upper bound C4 for the uncertainty product.
    coeffs = [a0, a4, a8], a12 is derived from g(0)=0.
    
    For g(x) = sum a_n H_n(x) exp(-x^2/2), we need to compute
    the uncertainty measure. The bound involves the ratio of norms
    and moment calculations.
    """
    a0, a4, a8 = coeffs
    # a12 from g(0) = 0: a0*1 + a4*3 + a8*105 + a12*10395 = 0
    a12 = -(a0 + 3*a4 + 105*a8) / 10395.0
    
    # Evaluate g on a fine grid for numerical integration
    x = np.linspace(-10, 10, 20001)
    dx = x[1] - x[0]
    
    gauss = np.exp(-x**2 / 2)
    H0 = np.ones_like(x)
    H4 = 16 * x**4 - 48 * x**2 + 12
    H8 = 256 * x**8 - 1792 * x**6 + 3360 * x**4 - 1680 * x**2 + 168
    H12 = (4096 * x**12 - 46080 * x**10 + 241920 * x**8 
           - 483840 * x**6 + 332640 * x**4 - 70560 * x**2 + 10395)
    
    g = (a0 * H0 + a4 * H4 + a8 * H8 + a12 * H12) * gauss
    
    # Check g(0) is approximately 0
    g0 = a0 + 3*a4 + 105*a8 + a12*10395
    if abs(g0) > 1e-8:
        return 1e10
    
    # Norm squared: ||g||^2 = integral |g|^2 dx
    norm_sq = np.trapz(g**2, x)
    if norm_sq < 1e-15:
        return 1e10
    
    # For the uncertainty bound, we need:
    # A(f) related to spread in x-space, A(F(f)) related to spread in frequency
    # The product bound for Hermite-Gaussian combinations can be computed
    # via the ratio of weighted norms.
    
    # The uncertainty principle bound: (var_x)(var_k) >= 1/4
    # For the specific inequality with C4, we compute:
    # C4 = sup over test functions of the ratio
    
    # Numerical approach: compute the moments
    # var_x = <x^2> - <x>^2
    # var_k = <k^2> - <k>^2 where k is Fourier variable
    
    # For g(x) = h(x) * exp(-x^2/2), Fourier transform is also Gaussian-weighted
    # Use the fact that for Hermite functions, the Fourier transform maps H_n to (-i)^n H_n
    
    # Compute <x>, <x^2> in x-space
    x_mean = np.trapz(x * g**2, x) / norm_sq
    x2_mean = np.trapz(x**2 * g**2, x) / norm_sq
    var_x = x2_mean - x_mean**2
    
    # For Fourier space, use that F[exp(-x^2/2) H_n(x)] = (-i)^n sqrt(2*pi) exp(-k^2/2) H_n(k)
    # So if g(x) = sum c_n exp(-x^2/2) H_n(x), then 
    # G(k) = sqrt(2*pi) sum c_n (-i)^n exp(-k^2/2) H_n(k)
    # |G(k)|^2 = 2*pi * |sum c_n (-i)^n exp(-k^2/2) H_n(k)|^2
    
    # Compute in Fourier space
    G = (a0 * H0 + a4 * (-1)**2 * H4 + a8 * (-1)**4 * H8 + a12 * (-1)**6 * H12) * gauss
    # Note: (-i)^0=1, (-i)^4=1, (-i)^8=1, (-i)^12=1 for even n
    # Actually (-i)^4 = 1, (-i)^8 = 1, (-i)^12 = 1
    # So for even degrees, the Fourier transform has the same form
    
    G_norm_sq = np.trapz(G**2, x)
    if G_norm_sq < 1e-15:
        return 1e10
    
    k_mean = np.trapz(x * G**2, x) / G_norm_sq
    k2_mean = np.trapz(x**2 * G**2, x) / G_norm_sq
    var_k = k2_mean - k_mean**2
    
    # The uncertainty product
    if var_x < 1e-15 or var_k < 1e-15:
        return 1e10
    
    # C4 bound is related to the product of spreads
    # The inequality states A(f)*A(F(f)) <= C4 * ||f|| * ||F(f)||
    # We want to minimize this bound
    
    # A common formulation: the bound is proportional to sqrt(var_x * var_k) / (||g|| * ||G||)
    # But let's use the direct product of standard deviations as the measure
    uncertainty_product = np.sqrt(var_x * var_k)
    
    return uncertainty_product

def _objective(coeffs):
    """Objective function for optimization."""
    try:
        result = _compute_c4_bound(coeffs)
        if not np.isfinite(result):
            return 1e10
        return result
    except:
        return 1e10

def construct():
    """Find optimal coefficients for the Hermite combination."""
    start_time = time.time()
    deadline = start_time + 180.0  # Safety margin before 210s kill
    
    # Initial guess from the seed
    best_coeffs = np.array([-113.0 / 100.0, 1.0 / 25.0, 1.0 / 3240.0])
    best_score = _compute_c4_bound(best_coeffs)
    
    # Try multiple starting points with local optimization
    start_points = [
        best_coeffs.copy(),
        np.array([-1.0, 0.05, 0.0003]),
        np.array([-1.2, 0.04, 0.0002]),
        np.array([-0.9, 0.06, 0.0004]),
        np.array([-1.1, 0.03, 0.0005]),
        np.array([-1.05, 0.045, 0.00035]),
    ]
    
    for sp in start_points:
        if time.time() > deadline - 30.0:
            break
        try:
            result = minimize(
                _objective,
                sp,
                method='Nelder-Mead',
                options={'maxiter': 5000, 'xatol': 1e-12, 'fatol': 1e-12, 'adaptive': True}
            )
            if result.fun < best_score:
                best_score = result.fun
                best_coeffs = result.x.copy()
        except:
            pass
    
    # Also try a broader search with differential evolution if time permits
    if time.time() < deadline - 60.0:
        try:
            bounds = [(-2.0, 0.0), (-0.5, 0.5), (-0.01, 0.01)]
            de_result = differential_evolution(
                _objective,
                bounds=bounds,
                maxiter=100,
                tol=1e-10,
                seed=42,
                polish=True,
                workers=1
            )
            if de_result.fun < best_score:
                best_score = de_result.fun
                best_coeffs = de_result.x.copy()
        except:
            pass
    
    # Final polish with tight tolerance
    if time.time() < deadline - 10.0:
        try:
            polish_result = minimize(
                _objective,
                best_coeffs,
                method='Nelder-Mead',
                options={'maxiter': 10000, 'xatol': 1e-14, 'fatol': 1e-14, 'adaptive': True}
            )
            if polish_result.fun < best_score:
                best_score = polish_result.fun
                best_coeffs = polish_result.x.copy()
        except:
            pass
    
    return np.array(best_coeffs)
# EVOLVE-BLOCK-END
