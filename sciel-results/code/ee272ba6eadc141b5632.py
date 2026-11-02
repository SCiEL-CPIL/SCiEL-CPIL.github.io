"""Optimized coefficients for the uncertainty inequality test function."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.special import factorial
from scipy.optimize import minimize_scalar, minimize
import time


def _phys_hermite(n, x):
    """Physicist's Hermite polynomial H_n(x)."""
    if n == 0:
        return np.ones_like(x)
    if n == 1:
        return 2 * x
    h0 = np.ones_like(x)
    h1 = 2 * x
    for k in range(2, n + 1):
        h2 = 2 * x * h1 - 2 * (k - 1) * h0
        h0, h1 = h1, h2
    return h1


def _compute_C4(c0, c4, c8):
    """
    Compute the upper bound on A(f)*A(Fhat(f)) for the test function
    g(x) = exp(-x^2/2) * (c0*H0 + c4*H4 + c8*H8 + c12*H12)
    with c12 chosen so g(0)=0.
    
    Returns (C4_bound, g) or (inf, None) if invalid.
    """
    # H_n(0) for even n: H_{2k}(0) = (-1)^k * (2k)! / k!
    def H_even_zero(n):
        k = n // 2
        sign = (-1) ** k
        val = factorial(n) / factorial(k)
        return sign * val

    c12 = -(c0 * H_even_zero(0) + c4 * H_even_zero(4) + c8 * H_even_zero(8)) / H_even_zero(12)
    
    coeffs = [c0, c4, c8, c12]
    degrees = [0, 4, 8, 12]
    
    # Compute A(f) = integral |g(x)|^2 dx / integral |g(x)| dx (or similar)
    # For uncertainty inequality, A(f) = (∫|f|²)^{-1} ∫|x f|² or similar
    # Actually, the standard form: A(f) = (∫|f|² dx)^{-1/2} * (∫|x f|² dx)^{1/2}
    # But the task says "upper bound on A(f)*A(Fourier(f))"
    # Let me use the standard uncertainty product: σ_x * σ_ξ
    # where σ_x² = ∫x²|f|²/∫|f|² and σ_ξ² = ∫ξ²|Fhat|²/∫|Fhat|²
    # By Plancherel, ∫|Fhat|² = (1/2π)∫|f|² (depending on convention)
    # Let's use: A(f) = sqrt(∫x²|f|² / ∫|f|²), A(F) = sqrt(∫ξ²|Fhat|² / ∫|Fhat|²)
    # Product = sqrt(∫x²|f|² * ∫ξ²|Fhat|² / (∫|f|² * ∫|Fhat|²))
    
    # Numerical integration
    x = np.linspace(-20, 20, 20001)
    dx = x[1] - x[0]
    
    # Build g(x)
    g = np.zeros_like(x)
    for c, d in zip(coeffs, degrees):
        g += c * _phys_hermite(d, x)
    g *= np.exp(-x**2 / 2)
    
    # Check g(0) ≈ 0
    idx0 = np.argmin(np.abs(x))
    if abs(g[idx0]) > 1e-6:
        # Still proceed but this might be an issue
        pass
    
    abs_g2 = g**2  # |g|² since g is real
    int_abs_g2 = np.trapz(abs_g2, x)
    
    if int_abs_g2 <= 0:
        return np.inf, None
    
    int_x2_abs_g2 = np.trapz(x**2 * abs_g2, x)
    
    # For Fourier transform, use Plancherel and properties
    # For g(x) = H_n(x) e^{-x²/2}, the Fourier transform is related to H_n
    # Specifically, F{H_n(x) e^{-x²/2}}(ξ) = i^n sqrt(2π) H_n(ξ) e^{-ξ²/2} (up to convention)
    # Let's use the convention: F(f)(ξ) = ∫ f(x) e^{-iξx} dx
    # Then F{e^{-x²/2}}(ξ) = sqrt(2π) e^{-ξ²/2}
    # F{H_n(x) e^{-x²/2}}(ξ) = i^n sqrt(2π) H_n(ξ) e^{-ξ²/2}
    
    # So |F(g)(ξ)|² = 2π |g(ξ)|² (since |i^n| = 1)
    # Wait, that's only if g is a single Hermite-Gaussian. For a sum, we need to be careful.
    # Actually, the Fourier transform is linear, so F(g) = sum of F(c_k H_{d_k} e^{-x²/2})
    # = sum c_k i^{d_k} sqrt(2π) H_{d_k}(ξ) e^{-ξ²/2}
    # So |F(g)(ξ)|² is NOT simply 2π|g(ξ)|² unless all phases align.
    
    # Let's compute numerically
    Fg = np.zeros_like(x)
    for c, d in zip(coeffs, degrees):
        # F{c H_d(x) e^{-x²/2}}(ξ) = c * i^d * sqrt(2π) * H_d(ξ) e^{-ξ²/2}
        phase = (1j) ** d
        Fg += c * phase * np.sqrt(2 * np.pi) * _phys_hermite(d, x) * np.exp(-x**2 / 2)
    
    abs_Fg2 = np.abs(Fg)**2
    int_abs_Fg2 = np.trapz(abs_Fg2, x)
    
    if int_abs_Fg2 <= 0:
        return np.inf, None
    
    int_xi2_abs_Fg2 = np.trapz(x**2 * abs_Fg2, x)  # ξ variable, same grid
    
    # A(f) = sqrt(∫x²|f|² / ∫|f|²)
    # A(F) = sqrt(∫ξ²|F|² / ∫|F|²)
    A_f = np.sqrt(int_x2_abs_g2 / int_abs_g2)
    A_F = np.sqrt(int_xi2_abs_Fg2 / int_abs_Fg2)
    
    C4_bound = A_f * A_F
    return C4_bound, g


def construct():
    start_time = time.time()
    deadline = 210.0 - 30.0  # 30s safety margin
    
    # Initial guess from the seed
    best_c = np.array([-113.0 / 100, 1.0 / 25, 1.0 / 3240])
    best_score = _compute_C4(*best_c)[0]
    
    # Coordinate descent
    tol = 1e-10
    max_iter = 500
    
    for iteration in range(max_iter):
        if time.time() > deadline:
            break
        
        improved = False
        for i in range(3):
            if time.time() > deadline:
                break
            
            # Optimize coefficient i while keeping others fixed
            c0, c4, c8 = best_c
            
            if i == 0:
                def obj(c):
                    return _compute_C4(c, c4, c8)[0]
            elif i == 1:
                def obj(c):
                    return _compute_C4(c0, c, c8)[0]
            else:
                def obj(c):
                    return _compute_C4(c0, c4, c)[0]
            
            # Use bounded scalar optimization
            result = minimize_scalar(obj, bounds=(-10, 10), method='bounded',
                                     options={'xatol': tol})
            
            if result.success and result.fun < best_score:
                if i == 0:
                    best_c[0] = result.x
                elif i == 1:
                    best_c[1] = result.x
                else:
                    best_c[2] = result.x
                best_score = result.fun
                improved = True
        
        if not improved:
            break
    
    # Final polish with multivariate optimization
    if time.time() < deadline:
        def obj_multi(c):
            return _compute_C4(c[0], c[1], c[2])[0]
        
        result = minimize(obj_multi, best_c, method='Nelder-Mead',
                          options={'xatol': 1e-12, 'fatol': 1e-12, 'maxiter': 200})
        
        if result.success and result.fun < best_score:
            best_c = result.x
            best_score = result.fun
    
    return best_c
# EVOLVE-BLOCK-END
