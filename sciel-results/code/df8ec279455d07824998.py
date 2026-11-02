"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.special import factorial
import time
import math


def _compute_bound(c0, c4, c8):
    """Compute the upper bound on A(f)*A(F) for given coefficients.
    
    Uses exact Hermite-Gaussian Fourier properties and Gauss-Hermite quadrature.
    """
    # Coefficients for H_n(0) to determine c12 such that g(0)=0
    H0_0 = 1.0
    H4_0 = 12.0
    H8_0 = 1680.0
    H12_0 = 665280.0
    
    c12 = -(c0 * H0_0 + c4 * H4_0 + c8 * H8_0) / H12_0
    
    # Coefficients array: [c0, c4, c8, c12]
    coeffs = np.array([c0, c4, c8, c12])
    degrees = [0, 4, 8, 12]
    
    # For the Fourier transform of H_n(x) * exp(-x^2/2):
    # The Fourier transform (with convention F(k) = integral f(x) exp(-i*k*x) dx)
    # of H_n(x) exp(-x^2/2) is (-i)^n H_n(x) exp(-x^2/2) * sqrt(2*pi)
    # Wait, let me be more careful. The standard result is:
    # If f(x) = H_n(x) exp(-x^2/2), then F(k) = sqrt(2*pi) * (-i)^n * H_n(k) * exp(-k^2/2)
    # But we need to check the normalization.
    #
    # Actually, the Fourier transform convention matters. Let's use:
    # F(k) = integral_{-inf}^{inf} f(x) exp(-i*k*x) dx
    # For f(x) = exp(-x^2/2), F(k) = sqrt(2*pi) * exp(-k^2/2)
    # For f(x) = H_n(x) exp(-x^2/2), F(k) = sqrt(2*pi) * (-i)^n * H_n(k) * exp(-k^2/2)
    #
    # So if g(x) = sum_n c_n H_n(x) exp(-x^2/2), then
    # G(k) = sqrt(2*pi) * sum_n c_n (-i)^n H_n(k) exp(-k^2/2)
    #
    # For n = 0, 4, 8, 12, we have (-i)^n = 1 for all of them (since n is divisible by 4).
    # So G(k) = sqrt(2*pi) * g(k)
    #
    # This means A(F) = sqrt(2*pi) * A(f) if we define A(h) = sup |x*h(x)|
    # Wait, that's not right. Let me re-examine.
    #
    # Actually, the uncertainty product is A(f) * A(F) where:
    # A(f) = sup_x |x * f(x)|
    # A(F) = sup_k |k * F(k)|
    #
    # If F(k) = sqrt(2*pi) * g(k) where g has the same form as f, then:
    # A(F) = sup_k |k * sqrt(2*pi) * g(k)| = sqrt(2*pi) * sup_k |k * g(k)| = sqrt(2*pi) * A(f)
    #
    # So the product is A(f) * A(F) = sqrt(2*pi) * (A(f))^2
    #
    # But wait, this assumes the same coefficients. Let me double-check.
    # The function is g(x) = (c0*H0 + c4*H4 + c8*H8 + c12*H12) * exp(-x^2/2)
    # Its Fourier transform is G(k) = sqrt(2*pi) * (c0*H0 + c4*H4 + c8*H8 + c12*H12) * exp(-k^2/2)
    # because (-i)^n = 1 for n in {0,4,8,12}.
    #
    # So indeed G(k) = sqrt(2*pi) * g(k), and A(F) = sqrt(2*pi) * A(f).
    # The product is sqrt(2*pi) * (A(f))^2.
    #
    # To minimize the product, we minimize A(f) = sup_x |x * g(x)|.
    #
    # Let's compute A(f) using Gauss-Hermite quadrature for efficiency.
    
    # Use Gauss-Hermite quadrature with sufficient points
    N = 200
    x_gh, w_gh = np.polynomial.hermite.hermgauss(N)
    
    # Compute Hermite polynomials at quadrature points
    x2 = x_gh ** 2
    H0 = np.ones_like(x_gh)
    H4 = 16*x2*x2 - 48*x2 + 12
    H8 = 256*x2*x2*x2*x2 - 4480*x2*x2*x2 + 26880*x2*x2 - 6720*x2 + 1680
    x4 = x2 * x2
    x6 = x4 * x2
    x8 = x4 * x4
    x10 = x8 * x2
    x12 = x6 * x6
    H12 = 4096*x12 - 129024*x10 + 1761280*x8 - 11942400*x6 + 37158400*x4 - 54432000*x2 + 47900160
    
    # g(x) at quadrature points
    g_vals = (c0*H0 + c4*H4 + c8*H8 + c12*H12) * np.exp(-x_gh**2 / 2.0)
    
    # A(f) = sup |x * g(x)|. We need to find the maximum of |x * g(x)|.
    # Since we're using quadrature points, we approximate the sup by the max over these points.
    # But we should also check that we have enough points to capture the maximum.
    # The function x*g(x) is smooth and decays, so Gauss-Hermite points should be sufficient.
    
    xg = x_gh * g_vals
    Af = np.max(np.abs(xg))
    
    # The product is sqrt(2*pi) * Af^2
    bound = math.sqrt(2 * math.pi) * Af * Af
    
    return bound


def construct():
    """Find optimal coefficients by numerical optimization."""
    start_time = time.time()
    deadline = 170.0  # Safety margin before 210s limit
    
    # Initial guess from prior state of the art
    x0 = np.array([-113.0/100.0, 1.0/25.0, 1.0/3240.0])
    
    def objective(coeffs):
        try:
            return _compute_bound(coeffs[0], coeffs[1], coeffs[2])
        except:
            return 1e10
    
    best_val = objective(x0)
    best_x = x0.copy()
    
    # Try multiple start points
    start_points = [
        x0,
        np.array([-1.1, 0.04, 0.001]),
        np.array([-1.2, 0.03, 0.0005]),
        np.array([-1.0, 0.05, 0.001]),
        np.array([-1.15, 0.035, 0.0008]),
    ]
    
    from scipy.optimize import minimize
    
    for sp in start_points:
        if time.time() > deadline - 20:
            break
        try:
            res = minimize(objective, sp, method='Nelder-Mead', 
                          options={'maxiter': 200, 'xatol': 1e-8, 'fatol': 1e-10, 'adaptive': True})
            if res.fun < best_val:
                best_val = res.fun
                best_x = res.x.copy()
        except:
            pass
    
    # Final polish with tighter tolerance if time permits
    if time.time() < deadline - 10:
        try:
            res = minimize(objective, best_x, method='Nelder-Mead', 
                          options={'maxiter': 500, 'xatol': 1e-10, 'fatol': 1e-12, 'adaptive': True})
            if res.fun < best_val:
                best_val = res.fun
                best_x = res.x.copy()
        except:
            pass
    
    return np.array(best_x)
# EVOLVE-BLOCK-END
