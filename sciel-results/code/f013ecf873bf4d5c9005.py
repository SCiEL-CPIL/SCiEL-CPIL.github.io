"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

_DEADLINE = time.time() + 180.0


def _hermite_coeffs():
    """Return coefficients for H0, H4, H8, H12 as polynomials in x (descending power)."""
    # H0(x) = 1
    H0 = [1.0]
    # H4(x) = x^4 - 6x^2 + 3
    H4 = [1.0, 0.0, -6.0, 0.0, 3.0]
    # H8(x) = x^8 - 28x^6 + 210x^4 - 420x^2 + 105
    H8 = [1.0, 0.0, -28.0, 0.0, 210.0, 0.0, -420.0, 0.0, 105.0]
    # H12(x) = x^12 - 84x^10 + 2100x^8 - 16800x^6 + 30240x^4 - 12096x^2 + 10395
    H12 = [1.0, 0.0, -84.0, 0.0, 2100.0, 0.0, -16800.0, 0.0, 30240.0, 0.0, -12096.0, 0.0, 10395.0]
    return H0, H4, H8, H12


def _poly_integrate_weighted(p1, p2, weight_exp):
    """
    Compute integral of p1(x)*p2(x)*exp(-weight_exp * x^2) dx from -inf to inf.
    p1, p2 are coefficient arrays in descending power order.
    Uses the fact that for even polynomials, integral of x^(2k) * exp(-a*x^2) = 
    sqrt(pi/a) * (2k-1)!! / (2a)^k for integer k.
    """
    # Convert to ascending order
    p1_asc = p1[::-1]
    p2_asc = p2[::-1]
    
    # Multiply the two polynomials (convolution)
    prod = np.convolve(p1_asc, p2_asc)
    
    # Now prod[k] is coefficient of x^k
    # We need integral of x^k * exp(-a*x^2) dx
    # Only even powers contribute
    a = weight_exp
    result = 0.0
    for k in range(len(prod)):
        if k % 2 == 0:
            n = k // 2
            # integral of x^(2n) * exp(-a*x^2) dx = sqrt(pi/a) * (2n-1)!! / (2a)^n
            # (2n-1)!! = product of odd numbers up to (2n-1)
            # For n=0: 1
            # For n=1: 1
            # For n=2: 1*3 = 3
            # etc.
            double_factorial = 1.0
            for i in range(1, 2*n, 2):
                double_factorial *= i
            val = np.sqrt(np.pi / a) * double_factorial / ((2*a) ** n)
            result += prod[k] * val
        # Odd powers integrate to 0
    return result


def _compute_bound(coeffs):
    """
    Compute the upper bound on A(f)*A(Fourier(f)) for given Hermite coefficients.
    coeffs = [a0, a4, a8]
    """
    H0, H4, H8, H12 = _hermite_coeffs()
    
    a0, a4, a8 = coeffs
    # Determine a12 from g(0) = 0
    # g(0) = a0*H0(0) + a4*H4(0) + a8*H8(0) + a12*H12(0) = 0
    # H0(0) = 1, H4(0) = 3, H8(0) = 105, H12(0) = 10395
    a12 = -(a0 * 1.0 + a4 * 3.0 + a8 * 105.0) / 10395.0
    
    # The test function is g(x) = a0*H0(x) + a4*H4(x) + a8*H8(x) + a12*H12(x)
    # We need to compute the relevant quantities for the uncertainty bound.
    
    # For the uncertainty principle with Gaussian weight, the key is:
    # A(f) involves ||f||_2 and ||x*f||_2 type quantities
    # The bound typically involves ratios of these norms.
    
    # Let's compute ||g||_2^2 = integral |g(x)|^2 * exp(-x^2/2) dx (or similar weight)
    # Actually, the standard Hermite function uses weight exp(-x^2/2) for the L2 norm.
    
    # Let me use a simpler approach: compute the bound numerically via quadrature
    # on a sufficiently large interval.
    
    x = np.linspace(-10, 10, 10000)
    dx = x[1] - x[0]
    
    H0_vals = np.ones_like(x)
    H4_vals = x**4 - 6*x**2 + 3
    H8_vals = x**8 - 28*x**6 + 210*x**4 - 420*x**2 + 105
    H12_vals = x**12 - 84*x**10 + 2100*x**8 - 16800*x**6 + 30240*x**4 - 12096*x**2 + 10395
    
    g = a0*H0_vals + a4*H4_vals + a8*H8_vals + a12*H12_vals
    
    # For the uncertainty principle, we typically look at:
    # Var(x) * Var(p) >= 1/4
    # where p = -i d/dx in appropriate units
    
    # The bound C4 is related to the product of spreads.
    # Let's compute the relevant moments.
    
    # Using weight exp(-x^2/2) for normalization
    weight = np.exp(-x**2 / 2)
    
    # Normalize g so that integral |g|^2 * weight = 1
    norm_sq = np.sum(g**2 * weight) * dx
    if norm_sq <= 0:
        return 1e10
    
    g_norm = g / np.sqrt(norm_sq)
    
    # Compute <x^2> = integral x^2 |g|^2 * weight / integral |g|^2 * weight
    x2_moment = np.sum(x**2 * g_norm**2 * weight) * dx
    
    # Compute <p^2> = integral |g'|^2 * weight / integral |g|^2 * weight
    # where p = -i d/dx, so |p g|^2 = |g'|^2
    dg = np.gradient(g_norm, dx)
    p2_moment = np.sum(dg**2 * weight) * dx
    
    # The uncertainty product is sqrt(x2_moment * p2_moment)
    # But we need the specific bound C4 which may involve higher moments or a specific formulation.
    
    # For now, let's use the standard Heisenberg uncertainty product
    uncertainty_product = np.sqrt(x2_moment * p2_moment)
    
    return uncertainty_product


def construct():
    # Initial guess from Goncalves et al. 2017
    x0 = np.array([-113.0 / 100.0, 1.0 / 25.0, 1.0 / 3240.0])
    
    best_x = x0.copy()
    best_val = _compute_bound(x0)
    
    # Try local optimization
    try:
        result = minimize(_compute_bound, x0, method='Nelder-Mead',
                         options={'maxiter': 5000, 'xatol': 1e-12, 'fatol': 1e-12,
                                  'adaptive': True})
        if result.fun < best_val:
            best_x = result.x
            best_val = result.fun
    except Exception:
        pass
    
    # Check deadline
    if time.time() > _DEADLINE - 10:
        return best_x
    
    # Try a few random restarts
    rng = np.random.default_rng(42)
    for i in range(20):
        if time.time() > _DEADLINE - 15:
            break
        x_rand = x0 + rng.normal(0, 0.1, 3)
        try:
            result = minimize(_compute_bound, x_rand, method='Nelder-Mead',
                             options={'maxiter': 2000, 'xatol': 1e-10, 'fatol': 1e-10,
                                      'adaptive': True})
            if result.fun < best_val:
                best_x = result.x
                best_val = result.fun
        except Exception:
            pass
    
    return best_x
# EVOLVE-BLOCK-END
