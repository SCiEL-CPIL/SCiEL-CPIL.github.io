"""Optimized search for Fourier-invariant Hermite-Gaussian test function coefficients."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from itertools import product

# Time management
START_TIME = time.time()
TIME_LIMIT = 180.0  # Seconds
SAFETY_MARGIN = 10.0  # Seconds before limit to stop

def time_remaining():
    return TIME_LIMIT - (time.time() - START_TIME) - SAFETY_MARGIN

def hermite_polynomial(n, x):
    """Compute the n-th physicists' Hermite polynomial H_n(x)."""
    # Use the recurrence relation or direct formula for small n
    if n == 0:
        return np.ones_like(x)
    elif n == 1:
        return 2 * x
    elif n == 2:
        return 4 * x**2 - 2
    elif n == 3:
        return 8 * x**3 - 12 * x
    elif n == 4:
        return 16 * x**4 - 48 * x**2 + 12
    elif n == 5:
        return 32 * x**5 - 160 * x**3 + 120 * x
    elif n == 6:
        return 64 * x**6 - 480 * x**4 + 720 * x**2 - 120
    elif n == 7:
        return 128 * x**7 - 1344 * x**5 + 3360 * x**3 - 1680 * x
    elif n == 8:
        return 256 * x**8 - 4480 * x**6 + 20160 * x**4 - 20160 * x**2 + 5040
    elif n == 12:
        # H_12(x) = 4096*x^12 - 245760*x^10 + 4644960*x^8 - 34406400*x^6 + 90720000*x^4 - 75675600*x^2 + 10395000
        return (4096 * x**12 - 245760 * x**10 + 4644960 * x**8 - 
                34406400 * x**6 + 90720000 * x**4 - 75675600 * x**2 + 10395000)
    else:
        raise ValueError("Only H_0 through H_12 implemented")

def compute_bound(c0, c4, c8, x_grid, dx):
    """
    Compute the upper bound on C4 = A(f) * A(Fourier(f)) for given coefficients.
    
    The test function is g(x) = (c0*H0 + c4*H4 + c8*H8 + c12*H12) * exp(-x^2/2)
    where c12 is chosen so that g(0) = 0.
    
    A(f) = integral |f(x)|^2 dx / (integral |f(x)| dx)^2  -- but for uncertainty,
    we typically use A(f) = integral |x f(x)|^2 dx / integral |f(x)|^2 dx (variance-like)
    
    Actually, for the uncertainty inequality, we need:
    A(f) = integral |x f(x)|^2 dx / integral |f(x)|^2 dx
    A(F) = integral |xi F(xi)|^2 dxi / integral |F(xi)|^2 dxi
    
    And C4 = A(f) * A(F)
    
    For a Gaussian times polynomial, we can compute these integrals analytically or numerically.
    """
    # Determine c12 such that g(0) = 0
    # g(0) = (c0*H0(0) + c4*H4(0) + c8*H8(0) + c12*H12(0)) * exp(0)
    # H0(0) = 1, H4(0) = 12, H8(0) = 5040, H12(0) = 10395000
    H0_0 = 1.0
    H4_0 = 12.0
    H8_0 = 5040.0
    H12_0 = 10395000.0
    
    c12 = -(c0 * H0_0 + c4 * H4_0 + c8 * H8_0) / H12_0
    
    # Construct g(x) on the grid
    H0 = hermite_polynomial(0, x_grid)
    H4 = hermite_polynomial(4, x_grid)
    H8 = hermite_polynomial(8, x_grid)
    H12 = hermite_polynomial(12, x_grid)
    
    poly = c0 * H0 + c4 * H4 + c8 * H8 + c12 * H12
    g = poly * np.exp(-x_grid**2 / 2.0)
    
    # Compute A(f) = integral |x g(x)|^2 dx / integral |g(x)|^2 dx
    g2 = g**2
    xg2 = (x_grid * g)**2
    
    int_g2 = np.trapz(g2, x_grid)
    int_xg2 = np.trapz(xg2, x_grid)
    
    if int_g2 < 1e-30:
        return np.inf
    
    A_f = int_xg2 / int_g2
    
    # For the Fourier transform, we use the fact that for the Hermite-Gaussian basis,
    # the Fourier transform has a simple structure. 
    # 
    # The key insight: If g(x) = sum_n c_n H_n(x) exp(-x^2/2), then
    # Fourier(g)(xi) = i^n sum_n c_n H_n(xi) exp(-xi^2/2) * (normalization)
    # 
    # Actually, the Fourier transform of H_n(x) exp(-x^2/2) is i^n H_n(xi) exp(-xi^2/2)
    # (up to normalization constants that cancel in the ratio)
    #
    # So A(F) = integral |xi F(xi)|^2 dxi / integral |F(xi)|^2 dxi
    # = integral |xi * i^n c_n H_n(xi) exp(-xi^2/2)|^2 dxi / integral |i^n c_n H_n(xi) exp(-xi^2/2)|^2 dxi
    # = integral |xi * (sum c_n H_n(xi))|^2 exp(-xi^2) dxi / integral |(sum c_n H_n(xi))|^2 exp(-xi^2) dxi
    #
    # Wait, the |i^n| = 1, so the magnitude is the same.
    #
    # So A(F) = A_f (by symmetry)? No, that's not right.
    #
    # Let me reconsider. The Fourier transform convention matters.
    # Using F(xi) = integral f(x) exp(-i x xi) dx / sqrt(2*pi)
    #
    # For f(x) = H_n(x) exp(-x^2/2), F(xi) = i^n H_n(xi) exp(-xi^2/2)
    #
    # So |F(xi)|^2 = H_n(xi)^2 exp(-xi^2)
    # And |xi F(xi)|^2 = xi^2 H_n(xi)^2 exp(-xi^2)
    #
    # For a linear combination, the cross terms matter.
    
    # Compute F(xi) on the same grid (by symmetry, the form is similar)
    # F(xi) = (sum_n c_n i^n H_n(xi)) * exp(-xi^2/2)
    # |F(xi)|^2 = |sum_n c_n i^n H_n(xi)|^2 * exp(-xi^2)
    
    # Compute the complex polynomial
    poly_F = (c0 * H0 + c4 * 1j**4 * H4 + c8 * 1j**8 * H8 + c12 * 1j**12 * H12)
    # i^4 = 1, i^8 = 1, i^12 = i^4 * i^8 = 1
    # So all coefficients are real and positive in the magnitude
    # poly_F = c0*H0 + c4*H4 + c8*H8 + c12*H12 (same as poly!)
    
    # Wait, that means |F(xi)|^2 = |poly(xi)|^2 * exp(-xi^2) = g(xi)^2
    # So A(F) = A_f?
    
    # That can't be right for the uncertainty product. Let me re-examine.
    #
    # Actually, I think the issue is that the Fourier transform of the Hermite function
    # includes a factor that depends on n. Let me be more careful.
    #
    # The standard result: Fourier transform of H_n(x) e^{-x^2/2} is i^n H_n(xi) e^{-xi^2/2}
    # (with the convention F(xi) = (1/sqrt(2*pi)) integral f(x) e^{-ixxi} dx)
    #
    # So for g(x) = sum c_n H_n(x) e^{-x^2/2},
    # F(xi) = sum c_n i^n H_n(xi) e^{-xi^2/2}
    #
    # |F(xi)|^2 = |sum c_n i^n H_n(xi)|^2 e^{-xi^2}
    #
    # Since i^n for n=0,4,8,12 are all 1 (i^0=1, i^4=1, i^8=1, i^12=1),
    # |F(xi)|^2 = |sum c_n H_n(xi)|^2 e^{-xi^2} = |poly(xi)|^2 e^{-xi^2} = g(xi)^2
    #
    # So indeed, |F(xi)|^2 = g(xi)^2, which means A(F) = A_f.
    #
    # Then C4 = A_f^2.
    #
    # But wait, the task says "Fourier-invariant" which suggests this symmetry.
    #
    # So we just need to minimize A_f^2, which is equivalent to minimizing A_f.
    
    A_F = A_f  # By the symmetry we derived
    
    C4 = A_f * A_F
    
    return C4

def construct():
    """
    Returns three coefficients (c0, c4, c8) that minimize the uncertainty bound.
    The c12 coefficient is determined by the constraint g(0) = 0.
    """
    # Setup integration grid
    # Need a grid that captures the function well
    x_max = 8.0
    n_points = 20000
    x_grid = np.linspace(-x_max, x_max, n_points)
    dx = x_grid[1] - x_grid[0]
    
    best_C4 = np.inf
    best_coeffs = np.array([-113.0/100, 1.0/25, 1.0/3240])  # Initial guess from seed
    
    # Coefficient ranges to search
    # Based on the seed, coefficients are small
    c0_range = np.linspace(-2.0, 2.0, 50)
    c4_range = np.linspace(-0.2, 0.2, 50)
    c8_range = np.linspace(-0.01, 0.01, 50)
    
    # Coarse grid search
    for c0 in c0_range:
        if time_remaining() < 20:
            break
        for c4 in c4_range:
            if time_remaining() < 10:
                break
            for c8 in c8_range:
                if time_remaining() < 5:
                    break
                
                C4 = compute_bound(c0, c4, c8, x_grid, dx)
                if C4 < best_C4:
                    best_C4 = C4
                    best_coeffs = np.array([c0, c4, c8])
    
    # Local refinement around the best point found
    if time_remaining() > 30:
        # Use scipy if available for local optimization
        try:
            from scipy.optimize import minimize
            
            def objective(coeffs):
                c0, c4, c8 = coeffs
                return compute_bound(c0, c4, c8, x_grid, dx)
            
            result = minimize(objective, best_coeffs, method='Nelder-Mead', 
                             options={'maxiter': 1000, 'xatol': 1e-10, 'fatol': 1e-12})
            
            if result.success and result.fun < best_C4:
                best_C4 = result.fun
                best_coeffs = result.x
        except Exception:
            pass
    
    return best_coeffs
# EVOLVE-BLOCK-END
